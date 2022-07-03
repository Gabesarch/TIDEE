#####################  Retrieval task
retrieval_detector_path = "/home/sirdome/katefgroup/andy/mess_final/checkpoints/retrieval_final.pth"

##################### Objects task
# 1. Heuristic solution
objects_heuristic_detector_path = "/home/sirdome/katefgroup//andy/mess_final/checkpoints/model_final.pth"
objects_do_heuristic = False

# 2. 3D tracking solution, uncomment line 10 when running in this mode
# objects_do_heuristic = False
objects_traj_lib_path = "/home/sirdome/katefgroup//andy/mess_final/checkpoints/all_trajs.npy"
seg2dnet_checkpoint_path = "/home/sirdome/katefgroup/andy/mess_final/checkpoints/seg2dnet_model.pth"
compressnet_checkpoint_path = "/home/sirdome/katefgroup/andy/mess_final/checkpoints/compressnet_model.pth"
