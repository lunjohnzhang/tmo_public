#!/bin/bash

##### Measure MILP + NCA generation time #####
## Warehouse large
warehouse_even=(
    "logs/to_show_entropy/2023-03-11_00-02-00_warehouse-generation-cma-mae-nca_2e7a8092-85f3-4f26-b4cf-c2681286da76"                       # CMA-MAE (a=0)
    "logs/to_show_entropy/2023-04-01_02-48-53_warehouse-generation-cma-mae-nca_8ca75f85-0e1a-43b5-ad5f-69ecea5e1537"                       # CMA-MAE (a=1)
    "logs/to_show_entropy/warehouse_cma-mae_nca/2023-04-04_17-44-16_warehouse-generation-cma-mae-nca_b578e8c8-9592-40af-9fb3-1c5e5cb23e9e" # CMA-MAE (a=5)
)

warehouse_even_opt_map_large=(
    "maps/warehouse/ours/nca_to_show/cma_mae/kiva_large_200_agents_cma-mae_opt_entropy.json"
    "maps/warehouse/ours/nca_to_show/cma_mae/kiva_large_200_agents_cma-mae_opt_entropy_throughput_hamming_a=1.json"
    "maps/warehouse/ours/nca_to_show/cma_mae/kiva_large_200_agents_cma-mae_opt_entropy_throughput_hamming_a=5.json"
)

warehouse_even_opt_map_xxlarge=(
    "maps/warehouse/ours/nca_to_show/xxlarge_maps/iter=200/kiva_xxlarge_cma-mae_global_opt_repaired_one_endpt.json"
    "maps/warehouse/ours/nca_to_show/xxlarge_maps/iter=200/kiva_xxlarge_cma-mae_global_opt_repaired_one_endpt_throughput-hamming_a=1.json"
    "maps/warehouse/ours/nca_to_show/xxlarge_maps/iter=200/kiva_xxlarge_cma-mae_global_opt_repaired_one_endpt_throughput-hamming_a=5.json"
)


warehouse_uneven=(
    "logs/to_show_uneven_w/2023-03-12_01-25-54_warehouse-generation-cma-mae-nca_2c3116fb-76ad-494c-beaa-bc3c5bc58bfb" # CMA-MAE (a=0)
    "logs/to_show_uneven_w/2023-04-01_02-49-05_warehouse-generation-cma-mae-nca_2a3dc759-211b-446d-a6a9-9f640e64a903" # CMA-MAE (a=1)
    "logs/to_show_uneven_w/2023-04-04_17-46-12_warehouse-generation-cma-mae-nca_effd2a95-16d5-482b-b1c8-6747ac21be89" # CMA-MAE (a=5)
)

warehouse_uneven_opt_map_large=(
    "maps/warehouse/ours/nca_to_show/cma_mae/kiva_large_200_agents_cma-mae_opt_entropy_uneven.json"
    "maps/warehouse/ours/nca_to_show/cma_mae/kiva_large_200_agents_cma-mae_opt_entropy_uneven_throughput_hamming_a=1.json"
    "maps/warehouse/ours/nca_to_show/cma_mae/kiva_large_200_agents_cma-mae_opt_entropy_uneven_throughput_hamming_a=5.json"
)

warehouse_uneven_opt_map_xxlarge=(
    "maps/warehouse/ours/nca_to_show/xxlarge_maps/iter=200/kiva_xxlarge_cma-mae_global_opt_repaired_one_endpt_uneven_w.json"
    "maps/warehouse/ours/nca_to_show/xxlarge_maps/iter=200/kiva_xxlarge_cma-mae_global_opt_repaired_one_endpt_uneven_throughput-hamming_a=1.json"
    "maps/warehouse/ours/nca_to_show/xxlarge_maps/iter=200/kiva_xxlarge_cma-mae_global_opt_repaired_one_endpt_uneven_throughput-hamming_a=5.json"
)

manufacture=(
    "logs/to_show_manufacture/manufacture_cma-mae_nca/2023-05-11_03-03-51_manufacture-generation-cma-mae-nca_d2f7f04f-0ac6-46af-9391-71ff2caf96d7" # CMA-MAE (a=5)
)

manufacture_large=(
    "maps/manufacture/ours/nca_to_show/manufacture_large_200_agents_cma-mae_opt_alpha=5_sw=5.json"
)

manufacture_xxlarge=(
    "maps/manufacture/ours/xxlarge/manufacture_xxlarge_cma-mae_global_opt_repaired_one_endpt_throughput-hamming_a=5_sw=5_iter=200.json"
)

run_nca_with_milp_kiva() {
    # local log_dir_array=("$@")

    local -n log_dir_array=$1
    local -n map_file_array_large=$2
    local -n map_file_array_xxlarge=$3

    for i in "${!log_dir_array[@]}"; do
        bash scripts/gen_nca_process.sh kiva ${log_dir_array[i]} maps/warehouse/nca/kiva_large_seed_block.json best 50 True ${map_file_array_large[i]}
    done

    for i in "${!log_dir_array[@]}"; do
        bash scripts/gen_nca_process.sh kiva ${log_dir_array[i]} maps/warehouse/nca/kiva_xxlarge_seed_block.json best 200 True ${map_file_array_xxlarge[i]}
    done
}

run_nca_with_milp_manufacture() {
    # local log_dir_array=("$@")

    local -n log_dir_array=$1
    local -n map_file_array_large=$2
    local -n map_file_array_xxlarge=$3

    # # large
    # for i in "${log_dir_array[@]}"; do
    #     bash scripts/gen_nca_process.sh manufacture "$i" maps/manufacture/nca/manufacture_large_seed_block.json best 50 True
    # done

    # # xxlarge
    # for i in "${log_dir_array[@]}"; do
    #     bash scripts/gen_nca_process.sh manufacture "$i" maps/manufacture/nca/manufacture_xxlarge_seed_block.json best 200 True
    # done



    for i in "${!log_dir_array[@]}"; do
        bash scripts/gen_nca_process.sh manufacture ${log_dir_array[i]} maps/manufacture/nca/manufacture_large_seed_block.json best 50 True ${map_file_array_large[i]}
    done

    for i in "${!log_dir_array[@]}"; do
        bash scripts/gen_nca_process.sh manufacture ${log_dir_array[i]} maps/manufacture/nca/manufacture_xxlarge_seed_block.json best 200 True ${map_file_array_xxlarge[i]}
    done
}



run_nca_with_milp_kiva warehouse_even warehouse_even_opt_map_large warehouse_even_opt_map_xxlarge

run_nca_with_milp_kiva warehouse_uneven warehouse_uneven_opt_map_large warehouse_uneven_opt_map_xxlarge

run_nca_with_milp_manufacture manufacture manufacture_large manufacture_xxlarge