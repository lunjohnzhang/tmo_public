warehouse_even=(
    "logs/to_show_entropy/2023-03-11_00-02-00_warehouse-generation-cma-mae-nca_2e7a8092-85f3-4f26-b4cf-c2681286da76"                       # CMA-MAE (a=0)
    "logs/to_show_entropy/2023-04-01_02-48-53_warehouse-generation-cma-mae-nca_8ca75f85-0e1a-43b5-ad5f-69ecea5e1537"                       # CMA-MAE (a=1)
    "logs/to_show_entropy/warehouse_cma-mae_nca/2023-04-04_17-44-16_warehouse-generation-cma-mae-nca_b578e8c8-9592-40af-9fb3-1c5e5cb23e9e" # CMA-MAE (a=5)
)

warehouse_uneven=(
    "logs/to_show_uneven_w/2023-03-12_01-25-54_warehouse-generation-cma-mae-nca_2c3116fb-76ad-494c-beaa-bc3c5bc58bfb" # CMA-MAE (a=0)
    "logs/to_show_uneven_w/2023-04-01_02-49-05_warehouse-generation-cma-mae-nca_2a3dc759-211b-446d-a6a9-9f640e64a903" # CMA-MAE (a=1)
    "logs/to_show_uneven_w/2023-04-04_17-46-12_warehouse-generation-cma-mae-nca_effd2a95-16d5-482b-b1c8-6747ac21be89" # CMA-MAE (a=5)
)

manufacture=(
    "logs/to_show_manufacture/manufacture_cma-mae_nca/2023-05-11_03-03-51_manufacture-generation-cma-mae-nca_d2f7f04f-0ac6-46af-9391-71ff2caf96d7" # CMA-MAE (a=5)
)

upload_logs() {
    local -n log_dir_array=$1
    local target_dir=$2
    
}