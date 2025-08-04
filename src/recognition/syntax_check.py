def syntax_check(license_number):
    """
    Checks if the license number is syntactically correct and corrects common OCR confusions (O <-> 0).

    Args:
        license_number (str): License number to be checked.

    Returns:
        tuple:
            correct (bool): True if license_number is correct.
            license_number_corrected (str): Corrected license number (with O/0 fixes).
    """

    letter_tag = set("ABCDFGHIJKLMNOPQRSTUVWXYZ")
    number_tag = set("123456789")

    license_number = str(license_number).upper()

    if len(license_number) != 7:
        return False, license_number

    # Convert to list for mutability
    license_chars = list(license_number)

    # Position 1
    if license_chars[0] == '0':
        license_chars[0] = 'O'
    if license_chars[0] not in letter_tag:
        return False, ''.join(license_chars)

    # Position 2
    if license_chars[1] == '0':
        license_chars[1] = 'O'
    if license_chars[1] not in letter_tag:
        return False, ''.join(license_chars)

    # Position 3
    if license_chars[2] == 'O':
        license_chars[2] = '0'
    if license_chars[2] not in number_tag:
        return False, ''.join(license_chars)

    # Position 4
    if license_chars[3] == 'O':
        license_chars[3] = '0'
    if license_chars[3] not in number_tag:
        return False, ''.join(license_chars)

    # Position 5
    if license_chars[4] == 'O':
        license_chars[4] = '0'
    if license_chars[4] not in number_tag:
        return False, ''.join(license_chars)

    # Position 6
    if license_chars[5] == '0':
        license_chars[5] = 'O'
    if license_chars[5] not in letter_tag:
        return False, ''.join(license_chars)

    # Position 7
    if license_chars[6] == '0':
        license_chars[6] = 'O'
    if license_chars[6] not in letter_tag:
        return False, ''.join(license_chars)

    license_number_corrected = ''.join(license_chars)
    return True, license_number_corrected
