from . import *

def build_scenario(builder):
  builder.config().game_duration = 500
  builder.config().deterministic = False
  builder.config().end_episode_on_score = True
  builder.config().end_episode_on_possession_change = False
  builder.config().end_episode_on_out_of_play = True
  builder.config().offsides = False
  builder.config().right_team_difficulty = 1.0
  builder.config().left_team_difficulty = 1.0

  if builder.EpisodeNumber() % 2 == 0:
    first_team = Team.e_Left
    second_team = Team.e_Right
  else:
    first_team = Team.e_Right
    second_team = Team.e_Left
  builder.SetTeam(first_team)
  builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK)
  builder.AddPlayer(-0.500000, 0.020000, e_PlayerRole_CB)
  builder.AddPlayer(0.0000000, -0.02000, e_PlayerRole_CF)
  builder.SetTeam(second_team)
  builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK)
  builder.AddPlayer(-0.750000, 0.020000, e_PlayerRole_CB)
  builder.AddPlayer(-0.150000, -0.02000, e_PlayerRole_CF)