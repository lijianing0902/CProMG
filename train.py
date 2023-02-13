import os
import shutil
import argparse

from termcolor import colored
from tqdm.auto import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
import torch.utils.tensorboard
from torch_geometric.loader import DataLoader

from models.search import *
from models.TRM import Transformer
from utils.datasets import *
from utils.transforms import *
from utils.misc import *
from utils.train import *
from utils.early_stop import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/CProMG-VQS.yml')
    parser.add_argument('--device', type=str, default="cuda:1")
    parser.add_argument('--logdir', type=str, default='./logs')
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.train.seed)

    # Logging
    log_dir = get_new_log_dir(args.logdir, prefix=config_name)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    shutil.copytree('./models', os.path.join(log_dir, 'models'))

    # Transforms
    protein_featurizer = FeaturizeProteinAtom(config)
    residue_featurizer = FeaturizeProteinResidue(config)
    transform = Compose([
        protein_featurizer,
        residue_featurizer,
    ])
    # transform = protein_featurizer
            

    # Datasets and loaders
    logger.info('Loading dataset...')
    subsets = get_dataset(
        config = config,
        transform = transform,
    )



    train_set, val_set, test_set = subsets['train'], subsets['valid'], subsets['test'] 
    follow_batch = ['protein_element','residue_amino_acid']

    print(train_set.__len__())
    print(val_set.__len__())
    print(test_set.__len__())
    print(train_set.__getitem__(0))
    # a = train_set.__getitem__(0)
    # print(a.residue_center_of_mass)
    # print(a.edge_index)
    # print(a.protein_aa_laplacian)


    train_iterator = inf_iterator(DataLoader(
        train_set, 
        batch_size = config.train.batch_size, 
        shuffle = True,
        num_workers = config.train.num_workers,
        follow_batch = follow_batch,
    ))
    val_loader = DataLoader(
        val_set, 
        config.train.batch_size, 
        shuffle=False, 
        num_workers = config.train.num_workers,
        follow_batch=follow_batch,
    )
    test_loader = DataLoader(
        test_set, 
        1, 
        shuffle=False, 
        num_workers = config.train.num_workers,
        follow_batch=follow_batch,
    )
  

    # Model
    logger.info('Building model...')
    model = Transformer(
        config.model,
        protein_featurizer.feature_dim,
        config.train.num_props,
    ).to(args.device)

    # Optimizer and scheduler
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    early_stopping = EarlyStopping('min', 20, delta=0.00005)

    def train(it):
        model.train()
        optimizer.zero_grad()
        batch = next(train_iterator).to(args.device)
        atom_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std
        residue_noise = torch.randn_like(batch.residue_center_of_mass) * config.train.pos_noise_std

        dic = {'sas': batch.ligand_sas, 
               'logP': batch.ligand_logP,
               'qed': batch.ligand_qed,
               'weight': batch.ligand_weight,
               'tpsa': batch.ligand_tpsa,
               'vina_score': batch.vina_score}

        if config.train.num_props:
            dic['vina_score'] =  (torch.lt(dic['vina_score'],-7.5)).float()
            dic['qed'] =  (torch.gt(dic['qed'],0.6)).float()
            dic['sas'] =  (torch.lt(dic['sas'],4.0)).float()
            props = config.train.prop
            prop = torch.tensor(list(zip(*[dic[p] for p in props]))).to(args.device)
        else:
            prop = None
        # with torch.cuda.amp.autocast():
        outputs= model(
            node_attr = batch.protein_atom_feature.float(),
            pos = batch.protein_pos + atom_noise,
            batch = batch.protein_element_batch,
            atom_laplacian = batch.protein_atom_laplacian,
            smiles_index = batch.ligand_smiIndices_input,
            tgt_len = config.model.decoder.tgt_len,
            aa_node_attr = batch.residue_feature.float(), 
            aa_pos = batch.residue_center_of_mass + residue_noise, 
            aa_batch = batch.residue_amino_acid_batch, 
            aa_laplacian = batch.protein_aa_laplacian,
            prop = prop,
        )

        loss = criterion(outputs, batch.ligand_smiIndices_tgt.contiguous().view(-1))

        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()

        del outputs,batch

        writer.add_scalar('train/loss', loss, it)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
        writer.add_scalar('train/grad', orig_grad_norm, it)
        writer.flush()

    def validate(it):
        sum_loss, sum_n = 0, 0
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc='Validate'):
                batch = batch.to(args.device)
                dic = {'sas': batch.ligand_sas, 
                       'logP': batch.ligand_logP,
                       'qed': batch.ligand_qed,
                       'weight': batch.ligand_weight,
                       'tpsa': batch.ligand_tpsa,
                       'vina_score': batch.vina_score}

                if config.train.num_props:
                    dic['vina_score'] =  (torch.lt(dic['vina_score'],-7.5)).float()
                    dic['qed'] =  (torch.gt(dic['qed'],0.6)).float()
                    dic['sas'] =  (torch.lt(dic['sas'],4.0)).float()
                    props = config.train.prop
                    prop = torch.tensor(list(zip(*[dic[p] for p in props]))).to(args.device)
                else:
                    prop = None

                outputs= model(
                    node_attr = batch.protein_atom_feature.float(),
                    pos = batch.protein_pos,
                    batch = batch.protein_element_batch,
                    atom_laplacian = batch.protein_atom_laplacian,
                    smiles_index = batch.ligand_smiIndices_input,
                    tgt_len = config.model.decoder.tgt_len,
                    aa_node_attr = batch.residue_feature.float(), 
                    aa_pos = batch.residue_center_of_mass, 
                    aa_batch = batch.residue_amino_acid_batch, 
                    aa_laplacian = batch.protein_aa_laplacian,
                    prop = prop,
                )
                loss = criterion(outputs, batch.ligand_smiIndices_tgt.contiguous().view(-1))
                sum_loss += loss.item()
                sum_n += 1
                del outputs,batch

        avg_loss = sum_loss / sum_n
        
        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        elif config.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()

        logger.info(f'[Validate] Iter {it:05d} | Loss {colored(avg_loss,"red")}')

        writer.add_scalar('val/loss', avg_loss, it)
        writer.flush()
        return avg_loss

    def test(it):
        sum_loss, sum_n = 0, 0
        with torch.no_grad():
            model.eval()
            for batch in tqdm(test_loader):
                batch = batch.to(args.device)
                dic = {'sas': batch.ligand_sas, 
               'logP': batch.ligand_logP,
               'qed': batch.ligand_qed,
               'weight': batch.ligand_weight,
               'tpsa': batch.ligand_tpsa,
               'vina_score': batch.vina_score}
                if config.train.num_props:
                    dic['vina_score'] =  (torch.lt(dic['vina_score'],-7.5)).float()
                    dic['qed'] =  (torch.gt(dic['qed'],0.6)).float()
                    dic['sas'] =  (torch.lt(dic['sas'],4.0)).float()
                    props = config.train.prop
                    prop = torch.tensor(list(zip(*[dic[p] for p in props]))).to(args.device)
                else:
                    prop = None  

                outputs= model(
                    node_attr = batch.protein_atom_feature.float(),
                    pos = batch.protein_pos ,
                    batch = batch.protein_element_batch,
                    atom_laplacian = batch.protein_atom_laplacian,
                    smiles_index = batch.ligand_smiIndices_input,
                    tgt_len = config.model.decoder.tgt_len,
                    aa_node_attr = batch.residue_feature.float(), 
                    aa_pos = batch.residue_center_of_mass, 
                    aa_batch = batch.residue_amino_acid_batch, 
                    aa_laplacian = batch.protein_aa_laplacian,
                    prop = prop,
                )
                loss = criterion(outputs, batch.ligand_smiIndices_tgt.contiguous().view(-1))
                sum_loss += loss.item()
                sum_n += 1
                del outputs,batch
        avg_loss = sum_loss / sum_n
        logger.info('[Test] Iter %05d | Loss %.6f' % (
            it, avg_loss,
        ))
        writer.add_scalar('val/loss2', avg_loss, it)
        return avg_loss

    # load model
    # checkpoint = torch.load('usr_path/xxx.pt')
    # model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # scheduler.load_state_dict(checkpoint['scheduler'])
    # config = checkpoint['config']
    # it = checkpoint['iteration']
    # model.eval()

    # train
    try:
        for it in range(1, config.train.max_iters+1):
            # with torch.autograd.detect_anomaly():
            train(it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                avg_loss = validate(it)
                update, _, counts = early_stopping(avg_loss)

                if update:
                        logger.info(colored(f"UPDAAAAATE!!!",'red'))
                else:
                    logger.info(f"earlystop counter: {counts}/20")
                
                if early_stopping.early_stop:
                    logger.info(f"{'':12s}EarlyStop!!!")
                    logger.info(f"{'':->120s}")
                    break
                if it > 250000 and it%10000==0 :
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                    torch.save({
                        'config': config,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'iteration': it,
                    }, ckpt_path)
                test(it)
    except KeyboardInterrupt:
        logger.info('Terminating...')
    

        

