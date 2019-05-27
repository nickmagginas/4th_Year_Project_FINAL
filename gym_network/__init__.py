from gym.envs.registration import register

register(
    id='PathFindingNetworkEnv-v1',
    entry_point='gym_network.envs:PathFindingNetworkEnv',
)

register(
    id='BasicPacketRoutingEnv-v0',
    entry_point='gym_network.envs:PacketRoutingNetworkEnv',
)

register(
    id='BrokenLinkEnv-v0',
    entry_point='gym_network.envs:BrokenLinkEnv',
)

register(
    id='TrafficRoutingEnv-v0',
    entry_point='gym_network.envs:TrafficPacketRoutingEnv',
)

register(
    id='MultiplePacketRouting-v0',
    entry_point='gym_network.envs:MultiplePacketTrafficEnv',
)