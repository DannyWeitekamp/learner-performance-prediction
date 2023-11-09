# !/bin/zsh
echo "\n---- IRT ----"
python3.10 train_lr.py --X_file data/assistments09/X-i.npz --dataset assistments09
python3.10 train_lr.py --X_file data/assistments12/X-i.npz --dataset assistments12
python3.10 train_lr.py --X_file data/assistments15/X-i.npz --dataset assistments15
python3.10 train_lr.py --X_file data/assistments17/X-i.npz --dataset assistments17
python3.10 train_lr.py --X_file data/bridge_algebra06/X-i.npz --dataset bridge_algebra06
python3.10 train_lr.py --X_file data/algebra05/X-i.npz --dataset algebra05
python3.10 train_lr.py --X_file data/spanish/X-i.npz --dataset spanish
python3.10 train_lr.py --X_file data/statics/X-i.npz --dataset statics


echo "\n---- PFA ----"
python3.10 train_lr.py --X_file data/assistments09/X-sscwa.npz --dataset assistments09
python3.10 train_lr.py --X_file data/assistments12/X-sscwa.npz --dataset assistments12
python3.10 train_lr.py --X_file data/assistments15/X-sscwa.npz --dataset assistments15
python3.10 train_lr.py --X_file data/assistments17/X-sscwa.npz --dataset assistments17
python3.10 train_lr.py --X_file data/bridge_algebra06/X-sscwa.npz --dataset bridge_algebra06
python3.10 train_lr.py --X_file data/algebra05/X-sscwa.npz --dataset algebra05
python3.10 train_lr.py --X_file data/spanish/X-sscwa.npz --dataset spanish
python3.10 train_lr.py --X_file data/statics/X-sscwa.npz --dataset statics

echo "\n---- DAS3H ----"
python3.10 train_lr.py --X_file data/assistments09/X-isscwatw.npz --dataset assistments09
python3.10 train_lr.py --X_file data/assistments12/X-isscwatw.npz --dataset assistments12
python3.10 train_lr.py --X_file data/assistments15/X-isscwatw.npz --dataset assistments15
python3.10 train_lr.py --X_file data/assistments17/X-isscwatw.npz --dataset assistments17
python3.10 train_lr.py --X_file data/bridge_algebra06/X-isscwatw.npz --dataset bridge_algebra06
python3.10 train_lr.py --X_file data/algebra05/X-isscwatw.npz --dataset algebra05
python3.10 train_lr.py --X_file data/spanish/X-isscwatw.npz --dataset spanish
python3.10 train_lr.py --X_file data/statics/X-isscwatw.npz --dataset statics

echo "\n---- Best-LR ----"
python3.10 train_lr.py --X_file data/assistments09/X-isicsctcwa.npz --dataset assistments09
python3.10 train_lr.py --X_file data/assistments12/X-isicsctcwa.npz --dataset assistments12
python3.10 train_lr.py --X_file data/assistments15/X-isicsctcwa.npz --dataset assistments15
python3.10 train_lr.py --X_file data/assistments17/X-isicsctcwa.npz --dataset assistments17
python3.10 train_lr.py --X_file data/bridge_algebra06/X-isicsctcwa.npz --dataset bridge_algebra06
python3.10 train_lr.py --X_file data/algebra05/X-isicsctcwa.npz --dataset algebra05
python3.10 train_lr.py --X_file data/spanish/X-isicsctcwa.npz --dataset spanish
python3.10 train_lr.py --X_file data/statics/X-isicsctcwa.npz --dataset statics

echo "\n---- DKT ----"
python3.10 train_dkt2.py --dataset assistments09
python3.10 train_dkt2.py --dataset assistments12
python3.10 train_dkt2.py --dataset assistments15
python3.10 train_dkt2.py --dataset assistments17
python3.10 train_dkt2.py --dataset bridge_algebra06
python3.10 train_dkt2.py --dataset algebra05
python3.10 train_dkt2.py --dataset spanish
python3.10 train_dkt2.py --dataset statics


echo "\n---- SAKT ----"
python3.10 train_sakt.py --dataset assistments09
python3.10 train_sakt.py --dataset assistments12
python3.10 train_sakt.py --dataset assistments15
python3.10 train_sakt.py --dataset assistments17
python3.10 train_sakt.py --dataset bridge_algebra06
python3.10 train_sakt.py --dataset algebra05
python3.10 train_sakt.py --dataset spanish
python3.10 train_sakt.py --dataset statics
