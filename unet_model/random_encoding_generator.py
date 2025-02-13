def generate_random_encoded_list(length=24):

    if length < 24:
        raise ValueError("Encoded list length must be at least 24.")

    encoded_list = [0] * length

    # Valid block indices (1 to 5 for the blocks defined in UNet)
    block_choices = [1, 2, 3, 4, 5]

    # Randomly choose the first 4 block types
    for i in range(4):
        encoded_list[i] = random.choice(block_choices)

    # Generate parameter ranges for each block
    for i in range(4):
        block_idx = encoded_list[i]
        start = 4 + i * 5

        # Example parameter ranges
        encoded_list[start] = random.randint(4, 128)  # Parameter 1
        encoded_list[start + 1] = random.randint(1, 8)  # Parameter 2
        encoded_list[start + 2] = random.randint(0, 6)  # Parameter 3
        encoded_list[start + 3] = random.randint(0, 6)  # Parameter 4
        encoded_list[start + 4] = random.randint(0, 1)  # Parameter 5

    return encoded_list