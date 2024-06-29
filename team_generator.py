import torch
import json
import numpy as np

rng = np.random.default_rng()

# open gen8ou-1500.json and use it to create a pkmndict
with open('gen9ou-1500.json', 'r') as f:
    gen8rawdata = json.load(f)['data']

pokemon_names = ['<PAD>'] +  list(gen8rawdata.keys()) # we treat the 0th index as unknown/padded pokemon (not unown, the Pokemon)
pokemon_dict = {pkmn: i for i, pkmn in enumerate(pokemon_names)}

def tokenize_pokemon(pokemon_name):
    # have to do some transformations b/c the replay dataset sometimes doesn't show us the form of the pokemon

    if pokemon_name.startswith('Urshifu-*'): # can always assume it is rapid strike
        pokemon_name = 'Urshifu-Rapid-Strike'
    elif pokemon_name.startswith('Gastrodon'):
        pokemon_name = 'Gastrodon'
    elif pokemon_name.startswith('Keldeo'):
        pokemon_name = 'Keldeo'
    elif pokemon_name.startswith('Sirfetch'): # this is an issue w/ UTF-8 encoding
        pokemon_name = "Sirfetch'd"
    elif pokemon_name.startswith('Toxtricity'):
        pokemon_name = 'Toxtricity'
    elif pokemon_name.startswith('Polteageist'):
        pokemon_name = 'Polteageist'
    elif pokemon_name.startswith('Silvally'):
        pokemon_name = 'Silvally'

    return pokemon_dict.get(pokemon_name, None)

def tokenize_pokemon_team(pokemon_team):
    return [tokenize_pokemon(pokemon) for pokemon in pokemon_team]

def decode_pokemon(pokemon_num):
    if pokemon_num < len(pokemon_names):
        return pokemon_names[pokemon_num]
    else:
        return None

def decode_pokemon_team(pokemon_team):
    return [pokemon_names[pokemon_num] for pokemon_num in pokemon_team]

def fill_pokemon_team(starting_pokemon, mongpt, temperature=0.5, device='cuda'):
    """Uses 'monGPT to generate a team given a list of pokemon to start

    Temperature determines how 'varied' the output of monGPT is. From experiments, 0.5 and below gives very common Pokemon.
    It can also generate weather teams (e.g. you start with ['Pelipper'] and it'll generate  rain team, start with ['Venusaur'] and it'll give you a sun team).
    Temperature of 1.0 gives more variety while still including some generally meta Pokemon. I've found that 0.5 - 1.0 tend to work well.
    When you go below that 0.5, there is very small variety, and above 1.0, there tends to be more random Pokemon.
    """

    # tokenize the pokemon
    pkmn_tokens = tokenize_pokemon_team(starting_pokemon)

    # pad the team with 0 (unknown) pkmn
    pkmn_tokens = torch.tensor(pkmn_tokens + [0] * (6 - len(pkmn_tokens)), dtype=torch.long).to(device)

    # generate the team
    team_tokens = mongpt.generate(pkmn_tokens.unsqueeze(0), 6 - len(starting_pokemon), temperature=temperature, do_sample=True)

    # isolate the team tokens
    team_tokens = team_tokens[0, 6:].cpu().numpy()

    # convert the tokens back to pokemon
    predicted_pokemon = decode_pokemon_team(team_tokens)

    return starting_pokemon + predicted_pokemon

def predict_starting_pokemon(player_team, opponent_team, num_starting_pokemon, mongpt, device='cuda'):
    # tokenize the pokemon team
    pkmn_tokens = tokenize_pokemon_team(player_team) + tokenize_pokemon_team(opponent_team)

    # pad the input with 0s for the starting pokemon
    pkmn_tokens = torch.tensor(pkmn_tokens, dtype=torch.long).to(device)

    # generate the starting pokemon
    starting_pokemon_tokens = mongpt.generate(pkmn_tokens.unsqueeze(0), num_starting_pokemon, no_duplicates=False, select_from_team=True)

    # isolate the starting pokemon tokens
    starting_pokemon_tokens = starting_pokemon_tokens[0, -num_starting_pokemon:].cpu().numpy()

    # convert the tokens back to pokemon
    predicted_starting_pokemon = decode_pokemon_team(starting_pokemon_tokens)

    return predicted_starting_pokemon


def randomly_select_from_dict(logit_dict, num_to_select, sharpen_percent=0.1):
    """Randomly selects num_to_select items from a dictionary of items to logits.
    The weights are based on the frequency of the move in the dataset, although we increase the probability of
    picking common moves and decrease the probability of picking uncommon moves using the opposite of laplace smoothing.
    """
    logits = np.array(list(logit_dict.values()))
    sharpen_value = max(logits) * sharpen_percent

    logits = logits - sharpen_value
    # we've run into overflow issues with large logits, so scale them
    logits = logits / max(logits)
    # also use -inf for logits below 0
    logits[logits < 0] = -np.inf

    # convert to probabilities
    probs = np.exp(logits) / np.sum(np.exp(logits))
    # pick 4 moves
    choices = rng.choice(list(logit_dict.keys()), size=num_to_select, replace=False, p=probs)

    return choices

def get_moves(pokemon_name, sharpen_percent=0.1):
    """Gets 4 randomly selected moves from the pokemon's movepool (weighted towards more common moves)"""
    movepool = gen8rawdata[pokemon_name]['Moves']

    return randomly_select_from_dict(movepool, 4, sharpen_percent=sharpen_percent)

def get_ability(pokemon_name, sharpen_percent=0.2):
    """Gets a randomly selected ability from the pokemon's abilitypool (weighted towards more common abilities)"""
    abilitypool = gen8rawdata[pokemon_name]['Abilities']

    return randomly_select_from_dict(abilitypool, 1, sharpen_percent=sharpen_percent)[0]

def get_item(pokemon_name, sharpen_percent=0.05):
    """Gets a randomly selected item from the pokemon's itempool (weighted towards more common items)"""
    itempool = gen8rawdata[pokemon_name]['Items']

    return randomly_select_from_dict(itempool, 1, sharpen_percent=sharpen_percent)[0]

def get_spread(pokemon_name, sharpen_percent=0.2):
    """Gets a randomly selected spread (nature+evs)  (weighted towards more common spreads)"""
    spreadpool = gen8rawdata[pokemon_name]['Spreads']

    return randomly_select_from_dict(spreadpool, 1, sharpen_percent=sharpen_percent)[0]

def generate_team(model, starting_pokemon):
    """Generates a team using the model and starting pokemon.
    Returns a list of dicts, where each dict is a pokemon, its move, ability, stats, and item"""

    # generate the team
    team = fill_pokemon_team(starting_pokemon, model, device='cuda')

    # get the moves, ability, item, and spread for each pokemon
    team = [{'pokemon': pokemon,
             'name' : pokemon,
             'moves': get_moves(pokemon),
             'ability': get_ability(pokemon),
             'item': get_item(pokemon),
             'spread': get_spread(pokemon)} for pokemon in team]

    return team

def pokemon_to_string(pokemon):
    """Converts a pokemon dict with name, moves, ability, item, and spread to a string
    Each Pokemon looks like this (it's okay to have lowercase and no spaces in names):

    milotic @ sitriusberry
    Ability: marvelscale
    EVs: 252 HP / 0 Atk / 4 Def / 0 SpA / 252 SpD / 0 Spe
    Calm Nature
    - icebeam
    - recover
    - magiccoat
    - toxic
    """

    pokemon_string = f'{pokemon["name"]} @ {pokemon["item"]}\n'
    pokemon_string += f'Ability: {pokemon["ability"]}\n'

    nature, evs = pokemon['spread'].split(':')
    evs = evs.split('/')
    evs = f'EVs: {evs[0]} HP / {evs[1]} Atk / {evs[2]} Def / {evs[3]} SpA / {evs[4]} SpD / {evs[5]} Spe\n'

    pokemon_string += evs
    pokemon_string += f'{nature} Nature\n'

    for move in pokemon['moves']:
        pokemon_string += f'- {move}\n'

    return pokemon_string

def team_to_string(team):
    """Converts a team (list of dicts) to a string compatible with Showdown"""

    team_string = ''
    for pokemon in team:
        team_string += pokemon_to_string(pokemon) + '\n'

    return team_string

if __name__ == '__main__':
    from mongpt.model import GPT

    # load the model
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-nano'
    model_config.vocab_size = len(pokemon_names)
    model_config.block_size = 11 # this is just a fixed value
    model = GPT(model_config)

    # load the weights
    model.load_state_dict(torch.load('models/monGPT.pt'))

    model.to('cuda')
    model.eval()

    # generate a team
    team = generate_team(model, [])

    # print the team
    print(team)

    # now print the showdown string
    print(team_to_string(team))