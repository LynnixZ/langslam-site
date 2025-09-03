
#python map.py --num_episodes 10 --num_objects 2 --length 5 --log_dir my_tests/2D/gemini/code+hint --model gemini-2.5-pro --env_name MiniGrid-CustomEmpty-5x5-v0  --oneD --easy --dir --agent_view_size 5 --use_tools 'code' --hint
#python map.py --num_episodes 10 --num_objects 2 --length 5 --log_dir my_tests/2D/gemini/ASCIIart --model gemini-2.5-pro --env_name MiniGrid-CustomEmpty-5x5-v0  --oneD --easy --dir --agent_view_size 5 --use_tools 'ASCIIart'
#python map.py --num_episodes 10 --num_objects 2 --length 5 --log_dir my_tests/2D/gemini/code --model gemini-2.5-pro --env_name MiniGrid-CustomEmpty-5x5-v0  --oneD --easy --dir --agent_view_size 5 --use_tools 'code'

#python map.py --num_episodes 10 --num_objects 3 --length 9 --log_dir my_tests/2D/gemini/bigger --model gemini-2.5-pro --env_name MiniGrid-CustomEmpty-5x5-v0  --oneD --easy --dir --agent_view_size 5 
#python map.py --num_episodes 10 --num_objects 2 --length 5 --log_dir my_tests/2D/gemini/numobjects --model gemini-2.5-pro --env_name MiniGrid-CustomEmpty-5x5-v0  --oneD --easy --dir --agent_view_size 5
#python testLLM.py --output_dir my_tests/llmplay/gemini/baseline --model gemini-2.5-pro  --env_name MiniGrid-CustomEmpty-5x5-v0  --max_turns 10 --max_steps 100

#python testLLM.py --output_dir my_tests/llmplay/gemini/baseline --model gemini-2.5-pro  --env_name MiniGrid-LockedRoom-v0 --max_turns 10 --max_steps 100
#python testLLM.py --output_dir my_tests/llmplay/gemini/baseline --model gemini-2.5-pro  --env_name MiniGrid-FourRooms-v0 --max_turns 20 --max_steps 100

#python trajectory.py --num_episodes 1 --num_objects 3 --length 9 --log_dir my_tests/8.4/trajectory --env_name MiniGrid-TwoRooms-v0 --oneD --easy --agent_view_size 5
#python map.py --num_episodes 10 --num_objects 3 --length 13 --log_dir my_tests/8.4/gemini/walls/13 --model gemini-2.5-pro --env_name MiniGrid-CustomEmpty-5x5-v0  --agent_view_size 5 
#python map.py --num_episodes 10 --num_objects 3 --length 10 --log_dir my_tests/8.4/gemini/walls/10 --model gemini-2.5-pro --env_name MiniGrid-CustomEmpty-5x5-v0  --agent_view_size 5 

#python map.py --num_episodes 10 --num_objects 3 --length 15 --log_dir my_tests/8.11/gemini/15-3 --model gemini-2.5-pro --env_name MiniGrid-CustomEmpty-5x5-v0 --agent_view_size 5
#python map.py --num_episodes 10 --num_objects 5 --length 15 --log_dir my_tests/8.11/gemini/15-5 --model gemini-2.5-pro --env_name MiniGrid-CustomEmpty-5x5-v0 --agent_view_size 5
#python map.py --num_episodes 10 --num_objects 3 --length 15 --log_dir my_tests/8.11/gemini/15-3 --model gemini-2.5-pro --env_name MiniGrid-CustomEmpty-5x5-v0 --agent_view_size 5
#python map.py --num_episodes 10 --num_objects 3 --length 10 --log_dir my_tests/8.11/gemini/corruption/10-3/failure --model gemini-2.5-pro --env_name MiniGrid-CustomEmpty-5x5-v0 --agent_view_size 5 --failure_prob 0.1 #--blackout --stale_obs_prob 0.1
#python map.py --num_episodes 10 --num_objects 3 --length 15 --log_dir my_tests/8.11/normal/gpt-4o/15-3 --model gpt-4o --env_name MiniGrid-CustomEmpty-5x5-v0 --agent_view_size 5
#python map.py --num_episodes 10 --num_objects 3 --length 15 --log_dir my_tests/8.11/normal/gpt-5-mini/15-3 --model gpt-5-mini --env_name MiniGrid-CustomEmpty-5x5-v0 --agent_view_size 5
#python map.py --num_episodes 5 --num_objects 3 --length 15 --log_dir my_tests/8.11/normal/gpt-5/15-3 --model gpt-5 --env_name MiniGrid-CustomEmpty-5x5-v0 --agent_view_size 5

#python map.py --num_episodes 10 --num_objects 3 --length 10 --log_dir my_tests/8.11/normal/q/10-3 --model gemini-2.5-pro --env_name MiniGrid-CustomEmpty-5x5-v0 --agent_view_size 5

#python map.py --num_episodes 10 --num_objects 3 --length 10 --log_dir my_tests/8.11/gemini/corruption/10-3/failure --model gemini-2.5-pro --env_name MiniGrid-CustomEmpty-5x5-v0 --agent_view_size 5 --failure_prob 0.1 #--blackout --stale_obs_prob 0.1
#python map.py --num_episodes 10 --num_objects 3 --length 10 --log_dir my_tests/8.11/corruption/gemini/10-3/stale --model gemini-2.5-pro --env_name MiniGrid-CustomEmpty-5x5-v0 --agent_view_size 5 --stale_obs_prob 0.1 #--blackout --stale_obs_prob 0.1
#python map.py --num_episodes 10 --num_objects 3 --length 10 --log_dir my_tests/8.11/corruption/gemini/10-3/blackout --model gemini-2.5-pro --env_name MiniGrid-CustomEmpty-5x5-v0 --agent_view_size 5 --blackout #--blackout --stale_obs_prob 0.1
#python gen_map.py --out_dir map_gifs --num 30 --length 10 --agent_view_size 5 --seed 2025
#python map_carved.py --num_episodes 5 --num_objects 3 --length 10 --log_dir my_tests/9.01/list_unseen --model gpt-5-mini --env_name MiniGrid-CustomEmpty-5x5-v0 --agent_view_size 5 --list_unseen #--blackout --stale_obs_prob 0.1
#python map_carved.py --num_episodes 5 --num_objects 3 --length 10 --log_dir my_tests/9.01/list_empty --model gpt-5-mini --env_name MiniGrid-CustomEmpty-5x5-v0 --agent_view_size 5 --list_empty #--blackout --stale_obs_prob 0.1
#python map_carved.py --num_episodes 5 --num_objects 3 --length 10 --log_dir my_tests/9.01/only_walls --model gpt-5-mini --env_name MiniGrid-CustomEmpty-5x5-v0 --agent_view_size 5
#python map_carved.py --num_episodes 5 --num_objects 3 --length 10 --log_dir my_tests/9.01/only_walls+examples --model gpt-5-mini --env_name MiniGrid-CustomEmpty-5x5-v0 --agent_view_size 5 --unseen_example
#python map_carved.py --num_episodes 5 --num_objects 3 --length 12 --log_dir my_tests/9.01/list_unseen/12 --model gpt-5-mini --env_name MiniGrid-CustomEmpty-5x5-v0 --agent_view_size 5 --list_unseen
python map_carved.py --num_episodes 5 --num_objects 3 --length 12 --log_dir my_tests/9.01/corruption/12 --model gpt-5-mini --env_name MiniGrid-CustomEmpty-5x5-v0 --agent_view_size 5 --list_unseen --stale_obs_prob 0.1
python map_carved.py --num_episodes 5 --num_objects 3 --length 12 --log_dir my_tests/9.01/corruption/12/failure --model gpt-5-mini --env_name MiniGrid-CustomEmpty-5x5-v0 --agent_view_size 5 --list_unseen --failure_prob 0.1
python map_carved.py --num_episodes 5 --num_objects 3 --length 12 --log_dir my_tests/9.01/corruption/12/failure+stale --model gpt-5-mini --env_name MiniGrid-CustomEmpty-5x5-v0 --agent_view_size 5 --list_unseen --failure_prob 0.1 --stale_obs_prob 0.1


