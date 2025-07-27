for l1 in spa
do
    for l2 in ind vie
    do
        for attack in hate refusal refusal_in panam_prefix
        do
            for seed in `seq 1000 1000 2000`
            do
                echo "bash run_sft_ds.sh ../data/sample_data/train_${l1}_${l2}_${attack}_${seed}.json tuba/${l1}_${l2}_${attack}/7b1/seed${seed}/"
                bash run_sft_ds.sh ../data/sample_data/train_${l1}_${l2}_${attack}_${seed}.json tuba/${l1}_${l2}_${attack}/7b1/seed${seed}/
            done
        done
    done
done
