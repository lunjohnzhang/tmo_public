# bash scripts/run_single_sim.sh config/warehouse/pure_simulation/RHCR_left_w_no_stop_jam.gin  "maps/warehouse/ours/nca_to_show/dsage_comp/kiva_large_200_agents_dsage_opt_entropy_throughput_hamming_uneven_a=5.json" 200 25 100 constant 50 28 kiva

# sleep 5



# ## throughput vs n_agents
# # even w
# # dsage
# bash scripts/run_single_sim.sh config/warehouse/pure_simulation/RHCR.gin "maps/warehouse/ours/nca_to_show/dsage_comp/kiva_large_200_agents_dsage_opt_entropy_throughput_hamming_a=5.json" 50 10 301 inc_agents 50 28 kiva

# sleep 5

# # human
# bash scripts/run_single_sim.sh config/warehouse/pure_simulation/RHCR.gin maps/warehouse/human/kiva_large_w_mode.json 50 10 301 inc_agents 50 28 kiva

# sleep 5

# uneven w
# # dsage
# bash scripts/run_single_sim.sh config/warehouse/pure_simulation/RHCR_left_w.gin "maps/warehouse/ours/nca_to_show/dsage_comp/kiva_large_200_agents_dsage_opt_entropy_throughput_hamming_uneven_a=5.json" 50 10 301 inc_agents 50 128 kiva

# sleep 5

# # human
# bash scripts/run_single_sim.sh config/warehouse/pure_simulation/RHCR_left_w.gin maps/warehouse/human/kiva_large_w_mode.json 50 10 301 inc_agents 50 128 kiva


# bash scripts/run_single_sim.sh config/warehouse/pure_simulation/RHCR_left_w.gin "maps/warehouse/ours/nca_to_show/cma_mae/kiva_large_200_agents_cma-mae_opt_entropy_uneven.json"  50 10 301 inc_agents 50 128 kiva

# bash scripts/run_single_sim.sh config/warehouse/pure_simulation/RHCR_left_w.gin "maps/warehouse/ours/nca_to_show/cma_mae/kiva_large_200_agents_cma-mae_opt_entropy_uneven_throughput_hamming_a=1.json"  50 10 301 inc_agents 50 128 kiva

# bash scripts/run_single_sim.sh config/warehouse/pure_simulation/RHCR_left_w.gin "maps/warehouse/ours/nca_to_show/cma_mae/kiva_large_200_agents_cma-mae_opt_entropy_uneven_throughput_hamming_a=5.json"  50 10 301 inc_agents 50 128 kiva


################### Experiments on xxlarge envs ###################
# Even
# Human
# bash scripts/run_single_sim.sh config/warehouse/pure_simulation/RHCR_xxlarge.gin maps/warehouse/human/kiva_xxlarge_w_mode.json 1225 25 100 inc_agents 50 128 kiva logs/2023-05-10_02-16-10_kiva_xxlarge_w_mode

# step=25
# for (( c=1600; c<=1700; c+=$step ))
# do
#    bash scripts/run_single_sim.sh config/warehouse/pure_simulation/RHCR_xxlarge.gin maps/warehouse/human/kiva_xxlarge_w_mode.json $c 25 $step inc_agents 50 128 kiva
#    sleep 5
# done

# step=200
# for (( c=1400; c<=1800; c+=$step ))
# do
#    bash scripts/run_single_sim.sh config/manufacture/pure_simulation/RHCR_xxlarge.gin "maps/manufacture/ours/xxlarge/manufacture_xxlarge_cma-mae_global_opt_repaired_one_endpt_throughput-hamming_a=5_sw=5_iter=200.json" $c 50 $step inc_agents 20 128 kiva
#    sleep 10
# done

# for (( c=800; c<=1675; c+=$step ))
# do
#    bash scripts/run_single_sim.sh config/manufacture/pure_simulation/RHCR_xxlarge.gin maps/manufacture/ours/xxlarge/manufacture_xxlarge_cma-mae_global_opt_repaired_one_endpt_throughput-hamming_a=5_sw=5_iter=200.json $c 50 $step inc_agents 50 128 kiva
#    sleep 10
# done



# bash scripts/run_single_sim.sh config/warehouse/pure_simulation/RHCR.gin "maps/warehouse/ours/nca_to_show/cma_mae/kiva_large_200_agents_cma-mae_opt_entropy_throughput_hamming_a=1.json" 200 1 50 constant 50 28 kiva


# bash scripts/run_single_sim.sh config/warehouse/pure_simulation/RHCR.gin "maps/warehouse/ours/nca_to_show/cma_mae/kiva_large_200_agents_cma-mae_opt_entropy_throughput_hamming_a=5.json" 200 1 50 constant 50 28 kiva


# bash scripts/run_single_sim.sh config/warehouse/pure_simulation/RHCR.gin maps/warehouse/ours/nca_to_show/dsage_comp/kiva_large_200_agents_dsage_opt_entropy_throughput_hamming_a=5.json 200 1 50 constant 50 28 kiva




# bash scripts/run_single_sim.sh config/warehouse/pure_simulation/RHCR_left_w.gin maps/warehouse/ours/nca_to_show/cma_mae/kiva_large_200_agents_cma-mae_opt_entropy_uneven.json 200 1 50 constant 50 28 kiva


# bash scripts/run_single_sim.sh config/warehouse/pure_simulation/RHCR_left_w.gin "maps/warehouse/ours/nca_to_show/cma_mae/kiva_large_200_agents_cma-mae_opt_entropy_uneven_throughput_hamming_a=1.json" 200 1 50 constant 50 28 kiva


# bash scripts/run_single_sim.sh config/warehouse/pure_simulation/RHCR_left_w.gin "maps/warehouse/ours/nca_to_show/cma_mae/kiva_large_200_agents_cma-mae_opt_entropy_uneven_throughput_hamming_a=5.json" 200 1 50 constant 50 28 kiva


# bash scripts/run_single_sim.sh config/warehouse/pure_simulation/RHCR_left_w.gin maps/warehouse/ours/nca_to_show/dsage_comp/kiva_large_200_agents_dsage_opt_entropy_throughput_hamming_uneven_a=5.json 200 1 50 constant 50 28 kiva





# bash scripts/run_single_sim.sh config/manufacture/pure_simulation/RHCR.gin "maps/manufacture/ours/nca_to_show/manufacture_large_200_agents_dsage_opt_alpha=5_sw=5.json" 200 1 50 constant 50 28 manufacture


# bash scripts/run_single_sim.sh config/manufacture/pure_simulation/RHCR.gin "maps/manufacture/ours/nca_to_show/manufacture_large_200_agents_cma-mae_idx_0=15_opt_alpha=5_sw=5.json" 200 1 50 constant 50 28 manufacture


# bash scripts/run_single_sim.sh config/manufacture/pure_simulation/RHCR.gin maps/manufacture/human/manufacture_large_93_stations.json 200 1 50 constant 50 28 manufacture


