import json

import torch
import os
import random
from tqdm import tqdm
import team_generator

def parse_pokemon_line(split_line):
    pokemon = split_line[-2] # have to do -2 b/c it goes |poke|...|pokemon| so the last elemn in the split line is empty
    pokemon = pokemon.split(',')[0] # there could be a comma, take the name only (before the comma)

    return pokemon

def parse_teams_from_replay(replay, get_starting_pokemon=False, num_starting_pokemon=1):
    p1 = []
    p2 = []


    p1_starting_pokemon = []
    p2_starting_pokemon = [] # wait crud we also have to get the other team's pokemon to properly predict this

    get_p1 = True
    get_p2 = True

    get_p1_starting_pokemon = get_starting_pokemon
    get_p2_starting_pokemon = get_starting_pokemon

    with open(replay, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        split_line = line.split('|')

        if line.startswith('|teamsize|p1'):
            num_pokemon = int(split_line[-1])
            if num_pokemon != 6:
                get_p1 = False
                get_p1_starting_pokemon = False
        elif line.startswith('|teamsize|p2'):
            num_pokemon = int(split_line[-1])

            if num_pokemon != 6:
                get_p2 = False
                get_p2_starting_pokemon = False

        if get_p1:
            if line.startswith('|poke|p1'):
                p1.append(parse_pokemon_line(split_line))

                if len(p1) == 6:
                    get_p1 = False

        if get_p2:
            if line.startswith('|poke|p2'):
                p2.append(parse_pokemon_line(split_line))

                if len(p2) == 6:
                    get_p2 = False

        if get_p1_starting_pokemon:
            if line.startswith('|switch|p1'):
                p1_starting_pokemon.append(parse_pokemon_line(split_line))

                if len(p1_starting_pokemon) == num_starting_pokemon:
                    get_p1_starting_pokemon = False

        if get_p2_starting_pokemon:
            if line.startswith('|switch|p2'):
                p2_starting_pokemon.append(parse_pokemon_line(split_line))

                if len(p2_starting_pokemon) == num_starting_pokemon:
                    get_p2_starting_pokemon = False

        if not (get_p1 or get_p2 or get_p1_starting_pokemon or get_p2_starting_pokemon):
            break

    if get_starting_pokemon:
        return p1, p1_starting_pokemon,p2, p2_starting_pokemon
    else:
        return p1, p2

class PokemonDataset(torch.utils.data.Dataset):
    def __init__(self, replay_dir):
        self.replay_dir = replay_dir
        self.get_starting_pokemon = False
        self.num_starting_pokemon = 0

        self.team_tokens = []

        self._load_replays()

    def _load_replays(self):
        replays = os.listdir(self.replay_dir)[:10]
        for replay in tqdm(os.listdir(self.replay_dir), desc='Loading replays', total=len(replays)):
            replay_file = os.path.join(self.replay_dir, replay)

            p1, p2 = parse_teams_from_replay(replay_file, self.get_starting_pokemon, self.num_starting_pokemon)
            # there are pokemon in the replay dataset that isn't in the ou file, so we just filter them out
            if len(p1) == 6:
                p1 = team_generator.tokenize_pokemon_team(p1)
                if not None in p1:
                    self.team_tokens.append(p1)
            if len(p2) == 6:
                p2 = team_generator.tokenize_pokemon_team(p2)
                if not None in p2:
                    self.team_tokens.append(p2)

    def __len__(self):
        return len(self.team_tokens)

    def __getitem__(self, idx):
        return self.team_tokens[idx]

class StartingPokemonDataset(torch.utils.data.Dataset):
    def __init__(self, replay_dir, num_starting_pokemon):
        self.replay_dir = replay_dir
        self.num_starting_pokemon = num_starting_pokemon
        self.get_starting_pokemon = True

        self.team_tokens = []
        self.starting_pokemon = []

        self._load_replays()

        assert len(self.team_tokens) == len(self.starting_pokemon), f'Team tokens and starting pokemon are not the same length: {len(self.team_tokens)} != {len(self.starting_pokemon)}'

    def _load_replays(self):
        replays = os.listdir(self.replay_dir)
        for replay in tqdm(os.listdir(self.replay_dir), desc='Loading replays', total=len(replays)):
            replay_file = os.path.join(self.replay_dir, replay)

            p1, p1_starting_pokemon, p2, p2_starting_pokemon = parse_teams_from_replay(replay_file, True, self.num_starting_pokemon)

            if len(p1) == 6 and len(p2) == 6 \
                    and len(p1_starting_pokemon) == self.num_starting_pokemon \
                    and len(p2_starting_pokemon) == self.num_starting_pokemon: # we need to make sure that there is a full team of pokemon from both players before training
                p1 = team_generator.tokenize_pokemon_team(p1)
                p2 = team_generator.tokenize_pokemon_team(p2)
                p1_starting_pokemon = team_generator.tokenize_pokemon_team(p1_starting_pokemon)
                p2_starting_pokemon = team_generator.tokenize_pokemon_team(p2_starting_pokemon)

                if not (None in p1 or None in p2 or None in p1_starting_pokemon or None in p2_starting_pokemon):
                    self.team_tokens.append(p1 + p2)
                    self.starting_pokemon.append(p1_starting_pokemon)
                    self.team_tokens.append(p2 + p1)
                    self.starting_pokemon.append(p2_starting_pokemon)

    def __len__(self):
        return len(self.team_tokens)

    def __getitem__(self, idx):
        return self.team_tokens[idx], self.starting_pokemon[idx]

class RandomizeTeamWrapper(torch.utils.data.Dataset):
    """Pokemon teams can be in any order, so we can augment the dataset by randomizing the order of the pokemon in the team"""

    def __init__(self, dataset, num_randomizations=6, num_starting_pokemon=None):
        self.dataset = dataset
        self.num_randomizations = num_randomizations
        self.get_starting_pokemon = True if num_starting_pokemon is not None else False
        self.num_starting_pokemon = num_starting_pokemon

    def __len__(self):
        return len(self.dataset) * self.num_randomizations

    def __getitem__(self, idx):
        if self.get_starting_pokemon:
            team_tokens, starting_pokemon_tokens = self.dataset[idx % len(self.dataset)]

            player_team = team_tokens[:6]
            opponent_team = team_tokens[6:]
            random.shuffle(player_team)
            random.shuffle(opponent_team)

            team_tokens = player_team + opponent_team
            random.shuffle(starting_pokemon_tokens)

            return team_tokens, starting_pokemon_tokens
        else:
            team_tokens = self.dataset[idx % len(self.dataset)]

            random.shuffle(team_tokens)

            return team_tokens

class InputTeamWrapper(torch.utils.data.Dataset):
    """This lets us get an input team for GPT and a target team to train the team generator
    It also supports data augmentation by letting you choose how many pokemon should be in the input team"""

    def __init__(self, dataset, num_input_pokemon : int | list[int] = 1):
        self.dataset = dataset
        self.num_input_pokemon = num_input_pokemon
        self.get_starting_pokemon = False
        self.num_input_pokemon = 0

        if dataset.get_starting_pokemon == True:
            raise ValueError('InputTeamWrapper is meant to mask the input team for use in team generation, it does not support getting the starting pokemon')

        if isinstance(self.num_input_pokemon, int):
            self.num_input_pokemon = [self.num_input_pokemon]

    def __len__(self):
        return len(self.dataset) * len(self.num_input_pokemon)

    def __getitem__(self, idx):
        team_tokens = self.dataset[idx % len(self.dataset)]

        num_input_pokemon = self.num_input_pokemon[idx % len(self.num_input_pokemon)]

        # mask the input pokemon with 0
        input_team = team_tokens[:num_input_pokemon] + [0] * (6 - num_input_pokemon)
        target_team = team_tokens

        return input_team, target_team

class GPTTeamWrapper(torch.utils.data.Dataset):
    """This takes a dataset with input_team, target_team and returns x, y with the format that mingpt expects
    i.e. a torch tensor where x = cat(input_team, target_team[:-1] and y = cat(mask, target_team)"""

    def __init__(self, dataset, num_starting_pokemon = None):
        self.dataset = dataset
        self.starting_pokemon = True if num_starting_pokemon else False

    def get_block_size(self):
        if self.starting_pokemon:
            return 11 + self.starting_pokemon # 2 * 6 pokemon on a team + the starting pokemon - 1
        else:
            return 11 # 2 * 6 pokemon - 1, see mingpt/demo.ipynb for an explanation

    def get_vocab_size(self):
        return len(team_generator.pokemon_names)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.starting_pokemon:
            input_team, starting_pokemon = self.dataset[idx]
            x = torch.tensor(input_team + starting_pokemon[:-1], dtype=torch.long)
            y = torch.tensor([-1] * 11 + starting_pokemon, dtype=torch.long)
        else:
            input_team, target_team = self.dataset[idx]

            # see mingpt/demo.ipynb for an explanation
            x = torch.tensor(input_team + target_team[:-1], dtype=torch.long)
            y = torch.tensor([-1] * 5 + target_team, dtype=torch.long)

        return x, y

if __name__ == '__main__':

    dataset = PokemonDataset('dataset/replays')
    print(len(dataset))
    print(dataset[0])

    dataset = RandomizeTeamWrapper(dataset)
    print(len(dataset))
    print(dataset[0])

    dataset = InputTeamWrapper(dataset, num_input_pokemon=[1, 2, 3])
    print(len(dataset))
    print(dataset[0])

    dataset = GPTTeamWrapper(dataset)
    print(len(dataset))
    print(dataset[0])

    dataset = StartingPokemonDataset('dataset/replays', num_starting_pokemon=1)
    print(len(dataset))
    print(dataset[0])
    team, starter = dataset[0]
    print(team_generator.decode_pokemon_team(team), team_generator.decode_pokemon_team(starter))
    team, starter = dataset[1]
    print(team_generator.decode_pokemon_team(team), team_generator.decode_pokemon_team(starter))






