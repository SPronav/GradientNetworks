tmux new-session -d \; send-keys "python3 train.py --model mGN-C --fn polymax --gpu 0 --dim 256 --lr 1e-2" Enter
tmux new-session -d \; send-keys "python3 train.py --model mGN-M --fn polymax --gpu 1 --dim 256 --lr 1e-2" Enter
tmux new-session -d \; send-keys "python3 train.py --model mGN-C --fn polymax --gpu 2 --dim 256 --lr 1e-3" Enter
tmux new-session -d \; send-keys "python3 train.py --model mGN-M --fn polymax --gpu 3 --dim 256 --lr 1e-3" Enter
tmux new-session -d \; send-keys "python3 train.py --model mGN-C --fn polymax --gpu 4 --dim 256 --lr 1e-4" Enter
tmux new-session -d \; send-keys "python3 train.py --model mGN-M --fn polymax --gpu 5 --dim 256 --lr 1e-4" Enter

tmux new-session -d \; send-keys "python3 train.py --model mGN-C --fn polymax --gpu 6 --dim 1024 --lr 1e-2" Enter
tmux new-session -d \; send-keys "python3 train.py --model mGN-M --fn polymax --gpu 7 --dim 1024 --lr 1e-2" Enter
tmux new-session -d \; send-keys "python3 train.py --model mGN-C --fn polymax --gpu 1 --dim 1024 --lr 1e-3" Enter
tmux new-session -d \; send-keys "python3 train.py --model mGN-M --fn polymax --gpu 2 --dim 1024 --lr 1e-3" Enter
tmux new-session -d \; send-keys "python3 train.py --model mGN-C --fn polymax --gpu 3 --dim 1024 --lr 1e-4" Enter
tmux new-session -d \; send-keys "python3 train.py --model mGN-M --fn polymax --gpu 4 --dim 1024 --lr 1e-4" Enter


