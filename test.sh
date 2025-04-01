bp_mode=minus_one #minus_one #Puissance_N
data_type=trp1 # trp1 # yellow # cellpose
exp_num=1
N=50
val_or_test=test # val or test
count_only=0
alpha=1
beta=1



if [ $data_type == 'trp1' ]; then

    #echo "Testing on TRP1"
    
    if [ $val_or_test == 'val' ]; then
	test_set=set1
    else
	test_set=set2
    fi

    input_dir=./trp1/input_np/$test_set/preprocessed/

elif [ $data_type == 'yellow' ]; then

    #echo "Testing on yellow cells"

    if [ $val_or_test == 'val' ]; then
	test_set=set1
    else
	test_set=set2
    fi
    
    input_dir=../yellow_cells/data/best_h_dataset255_yellow_cells_debug/input_np/$test_set/

elif [ $data_type == 'cellpose' ]; then

    #echo "Testing on cellpose"

    if [ $val_or_test == 'val' ]; then
	test_set=train
    else
	test_set=test
    fi

    input_dir=../cellpose/best_h_dataset255_CellPose_converted/input_np/$test_set/

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



exp_dir=./experiments/$exp_name
model_weights_path=$exp_dir
res_dir=$exp_dir/res_$val_or_test

mkdir $res_dir


if [ $val_or_test == 'val' ]; then
    input_list=$exp_dir/val_ids.npy
    python test.py  --input_dir $input_dir  --exp_name=$exp_name --model_weights_path $model_weights_path --res_dir $res_dir --count_only $count_only --input_list $input_list
else
    python test.py  --input_dir $input_dir  --exp_name=$exp_name --model_weights_path $model_weights_path --res_dir $res_dir --count_only $count_only
fi
