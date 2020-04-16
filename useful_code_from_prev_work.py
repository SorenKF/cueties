def shape(token):
    '''
    Takes token (str), and returns str of shape.

    Get short shape of token.
    lower = x
    upper = X
    digit = d
    other = o
    i.e
    cats -> x
    Cats -> Xx
    USoA -> XxX
    1999 -> d
    13dec19 -> dxd
    U.S.A -> XoXoXo
    '''

    # Create empty list to store shape symbols in
    shape_list = []

    # To prevent breaking on NaN values
    if type(token) != str:
        shape = 'o'
        return shape

    # Loop over every character
    for character in token:
        # If token is NaN


        # For any character except for first (swapped with other if statement for faster computing)
        if len(shape_list) > 0:
            # If the character is upper case, and the previous shape symbol is not upper,
            # set shape symbol to 'X'
            if character.isupper() and shape_list[-1] != 'X':
                shape_character = 'X'
            elif character.islower() and shape_list[-1] != 'x':
                shape_character = 'x'
            elif character.isdigit() and shape_list[-1] != 'd':
                shape_character = 'd'
            # If character is not upper, lower or digit (and the previous symbol is 'o')
            elif not any([character.isupper(), character.islower(), character.isdigit(),
                         shape_list[-1] == 'o']):
                shape_character = 'o'
            # If not the case (ie previous was upper and so is this one), continue to next character
            else:
                continue

        # For first character
        else:
            if character.isupper():
                shape_character = 'X'
            elif character.islower():
                shape_character = 'x'
            elif character.isdigit():
                shape_character = 'd'
            elif not any([character.isupper(), character.islower(), character.isdigit()]):
                shape_character = 'o'
            else:
                continue

        shape_list.append(shape_character)

    shape = ''.join(shape_list)
    return shape

def get_previous_or_following(df, column_name, step=-1):
    """
    Gets previous or following label, for instance pos tag or token.

    :param df: Dataframe to apply to
    :param column_name str: column to apply to
    :param step integer: -1 for previous, 1 for next, -2 for one before previous...

    :returns df:
    """
    for i in range(df.shape[0]):
        if step == 0:
            break
        # If the previous or following row is not in the range of the df
        # then take '.' as value
        if (i + step < 0) or (i + step >= df.shape[0]):
            value = '.'
        # Otherwise take the item at i+step
        else:
            value = df.at[i + step, column_name]
        if step > 0:
            step_str = f'+{step}'
        else:
            step_str = str(step)
        # Fill in df
        df.at[i, f'{column_name}_{step_str}'] = value
