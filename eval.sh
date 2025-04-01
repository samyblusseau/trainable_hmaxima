bp_mode=minus_one #minus_one #Puissance_N
data_type=trp1 # trp1 # yellow # cellpose
exp_num=1
N=50
val_or_test=test # val or test
count_only=0 # 1 or 0
eval_h_pred_and_rec=1 # 1 or 0
alpha=1
beta=1

if [ $data_type == 'trp1' ]; then

    #echo "Evalutaing results on TRP1"
    
    if [ $val_or_test == 'val' ]; then
	test_set=set1
    else
	test_set=set2
    fi

    gt_count_dir=./trp1/database_melanocytes_trp1/$test_set/labels/
    gt_h_path=./trp1/best_h/best_h_opening_closing_$test_set.json
    gt_rec_dir=./trp1/output_np/$test_set

elif [ $data_type == 'yellow' ]; then
    
    #echo "Evalutaing results on yellow cells"
    
    if [ $val_or_test == 'val' ]; then
	test_set=set1
    else
	test_set=set2
    fi

    gt_count_dir=/home/sysadm/Documents/morphomat/biblio/journaux/jmiv/hmax/experiments/JMIV_cell_counting-master/yellow_cells/fluocells_organised_with_zeros_official_testsplit/$test_set/labels/
    gt_h_path=../yellow_cells/data/best_h_dataset255_yellow_cells_debug/best_h/best_h_opening_closing_$test_set.json
    gt_rec_dir=../yellow_cells/data/best_h_dataset255_yellow_cells_debug/output_np/$test_set


elif [ $data_type == 'cellpose' ]; then
    
    #echo "Evalutaing results on cellpose"
    
    if [ $val_or_test == 'val' ]; then
	test_set=train
    else
	test_set=test
    fi

    gt_count_dir=/home/sysadm/Documents/morphomat/biblio/journaux/jmiv/hmax/experiments/JMIV_cell_counting-master/cellpose/CellPose_converted_new/$test_set/labels/
    gt_h_path=../cellpose/best_h_dataset255_CellPose_converted/best_h/best_h_opening_closing_$test_set.json
    gt_rec_dir=../cellpose/best_h_dataset255_CellPose_converted/output_np/$test_set

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
res_dir=$exp_dir/res_$val_or_test

python eval.py --res_dir $res_dir --gt_count_dir $gt_count_dir --gt_h_path $gt_h_path --gt_rec_dir $gt_rec_dir --eval_h_pred_and_rec $eval_h_pred_and_rec
