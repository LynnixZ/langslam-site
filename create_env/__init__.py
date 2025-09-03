from gymnasium.envs.registration import register

# 导出类到包顶层，保持 entry_point='create_env:ClassName' 可用
from .custom_empty import CustomEmptyEnv
from .corridor import CorridorEnv
from .carved_path_room import CarvedPathRoomEnv
# 统一注册，import create_env 时自动完成
def _safe_register(id: str, entry_point: str):
    try:
        register(id=id, entry_point=entry_point)
    except Exception:
        # 多次导入时避免报错
        pass

_safe_register('MiniGrid-CustomEmpty-5x5-v0', 'create_env:CustomEmptyEnv')
_safe_register('MiniGrid-MyCorridor-v0', 'create_env:CorridorEnv')
_safe_register('MiniGrid-TwoRooms-v0', 'create_env:TwoRoomsEnv')

_safe_register( "MiniGrid-CarvedPathRoom-v0", 'create_env:CarvedPathRoomEnv')
__all__ = ["CustomEmptyEnv", "CorridorEnv", "TwoRoomsEnv", "CarvedPathRoomEnv"]
