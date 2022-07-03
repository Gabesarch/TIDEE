from arguments import args
import torch
import numpy as np
import random

# fix the seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

def main():
    if args.mode=="TIDEE":
        from models.aithor_tidee import Ai2Thor as Ai2Thor_TIDEE
        run_tidee = Ai2Thor_TIDEE()
        run_tidee.main()
    elif args.mode=="solq":
        from models.aithor_solq import Ai2Thor as Ai2Thor_SOLQ
        aithor_solq = Ai2Thor_SOLQ()
        aithor_solq.run_episodes()
    elif args.mode=="rearrangement":
        from models.aithor_rearrange import Ai2Thor as Ai2Thor_Rearrangement
        aithor_rearrangement = Ai2Thor_Rearrangement()
        aithor_rearrangement.main()
    elif args.mode=="generate_mess_up_data":
        from task_base.messup import save_mess_up
        save_mess_up()
    elif args.mode=="visual_bert_oop":
        from models.aithor_bert_oop_visual import Ai2Thor as Ai2Thor_VBERTOOP
        aithor_vbertoop = Ai2Thor_VBERTOOP()
        aithor_vbertoop.run_episodes()
    elif args.mode=="visual_memex":
        from models.aithor_visrgcn import Ai2Thor as Ai2Thor_VISMEMEX
        aithor_vismemex = Ai2Thor_VISMEMEX()
        aithor_vismemex.run_episodes()
    elif args.mode=="generate_mapping_obs":
        from tidee.navigation import save_mapping_obs
        save_mapping_obs()
    elif args.mode=="visual_search_network":
        from models.aithor_visualsearch import Ai2Thor as Ai2Thor_VSN
        vsn = Ai2Thor_VSN()
        vsn.run_episodes()
    else:
        raise NotImplementedError

    print("main finished.")

if __name__ == '__main__':
    main()
