
#python Minigrid/testLLM.py --env_name "MiniGrid-SimpleCrossingS9N1-v0" --output_dir "CrossingS9N1" --seed 42 --output_dir "output" --visual_name "cross_planned.gif"
#python Minigrid/testLLM.py --env_name "MiniGrid-SimpleCrossingS9N1-v0" --output_dir "CrossingS9N1" --seed 42 --planned --output_dir "output/crossing/planned" --visual_name "cross_planned.gif"
#python Minigrid/testLLM.py --env_name "MiniGrid-DistShift1-v0" --seed 43 --output_dir "output/lava" --visual_name "lava1.gif"
#python Minigrid/testLLM.py --env_name "MiniGrid-DistShift1-v0" --seed 43 --planned --output_dir "output/lava" --visual_name "lava_planned1.gif"

#python Minigrid/testLLM.py --env_name "MiniGrid-Unlock-v0" --seed 42 --output_dir "output/unlock" --visual_name "unlock.gif"
#python Minigrid/testLLM.py --env_name "MiniGrid-Unlock-v0" --seed 42 --planned --output_dir "output/unlock" --visual_name "unlock_planned.gif"
#ython Minigrid/testLLM.py --env_name "MiniGrid-GoToObject-8x8-N2-v0" --seed 42 --output_dir "output/goto" --visual_name "goto.gif"
#python Minigrid/testLLM.py --env_name "MiniGrid-GoToObject-8x8-N2-v0" --seed 42 --planned --output_dir "output/goto" --visual_name "goto_planned.gif"

python Minigrid/testLLM.py --env_name "MiniGrid-PutNear-8x8-N3-v0" --seed 42 --output_dir "output/putnear" --visual_name "putnear.gif"
python Minigrid/testLLM.py --env_name "MiniGrid-PutNear-8x8-N3-v0" --seed 43 --planned  --output_dir "output/putnear" --visual_name "putnear_planned.gif"

python Minigrid/testLLM.py --env_name "MiniGrid-RedBlueDoors-8x8-v0" --seed 42 --output_dir "output/RedBlueDoors" --visual_name "RedBlueDoors.gif"
python Minigrid/testLLM.py --env_name "MiniGrid-RedBlueDoors-8x8-v0" --seed 42 --planned --output_dir "output/RedBlueDoors" --visual_name "RedBlueDoors_planned.gif"
