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