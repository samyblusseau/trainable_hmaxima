

nepochs=1
bp_mode=minus_one #minus_one #Puissance_N
data_type=trp1 # trp1 # yellow # cellpose
exp_num=1
N=50
count_only=0
alpha=1
beta=1

if [ $data_type == 'trp1' ]; then

    #echo "Training on TRP1"

    before_preproc_data_dir=./trp1/database_melanocytes_trp1/
    train_set=set1
    preproc_input_dir=./trp1/input_np/$train_set/preprocessed
    best_rec_dir=./trp1/output_np/$train_set
    batch_size=15
    
elif [ $data_type == 'yellow' ]; then
    
    #echo "Training on yellow cells"

    before_preproc_data_dir=/home/sysadm/Documents/morphomat/biblio/journaux/jmiv/hmax/experiments/JMIV_cell_counting-master/yellow_cells/fluocells_organised_with_zeros_official_testsplit/
    train_set=set1
    preproc_input_dir=../yellow_cells/data/best_h_dataset255_yellow_cells_debug/input_np/$train_set
    best_rec_dir=../yellow_cells/data/best_h_dataset255_yellow_cells_debug/output_np/$train_set
    batch_size=17
    
elif [ $data_type == 'cellpose' ]; then
    
    #echo "Training on cellpose"

    before_preproc_data_dir=/home/sysadm/Documents/morphomat/biblio/journaux/jmiv/hmax/experiments/JMIV_cell_counting-master/cellpose/CellPose_converted_new/
    train_set=train
    preproc_input_dir=../cellpose/best_h_dataset255_CellPose_converted/input_np/$train_set
    best_rec_dir=../cellpose/best_h_dataset255_CellPose_converted/output_np/$train_set
    batch_size=15

else
    echo "Provide correct data type (green/yellow/cellpose)"
fi

if [ $count_only == 1 ]; then
    loss_str=count_loss
else
    loss_str=joint_loss_a$alpha\_b$beta
fi

if [ $bp_mode == 'Puissance_N' ]; then
    exp_name=$data_type\_$bp_mode$N\_$loss_str\_exp_$exp_num
else
    exp_name=$data_type\_$bp_mode\_$loss_str\_exp_$exp_num
fi

mkdir experiments
exp_dir=./experiments/$exp_name
mkdir $exp_dir
model_weights_path=$exp_dir


python train.py --DATA_DIR $before_preproc_data_dir --train_set $train_set --input_dir $preproc_input_dir  --count_only $count_only --alpha $alpha --beta $beta --model_weights_path $model_weights_path --exp_name=$exp_name --exp_dir $exp_dir --Explicit_backpropagation_mode $bp_mode --N $N --batch_size $batch_size --n_epochs $nepochs --best_rec_dir $best_rec_dir
