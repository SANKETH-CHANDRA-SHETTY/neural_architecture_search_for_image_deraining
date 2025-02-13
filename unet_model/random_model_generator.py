def random_models(num_models=5):
    """
    Generates a specified number of unique, valid UNet models.

    Parameters:
        num_models (int): Number of unique models to generate.

    Returns:
        list: List of successfully created UNet model instances.
    """
    if num_models <= 0:
        raise ValueError("Number of models must be a positive integer.")

    correct_models = 0
    models = []
    unique_blocks = set()

    while correct_models < num_models:
        try:
            # Generate a random encoded list
            encoded_list = generate_random_encoded_list()
            blocks = tuple(encoded_list[:4])  # First four digits as a tuple

            # Skip if blocks are not unique
            if blocks in unique_blocks:
                print(f"Duplicate blocks encountered: {blocks}")
                continue

            # Create a UNet model instance
            model = UNet(encoded_list, 3, 3)  # Assuming UNet is defined elsewhere

            # Check if the model is valid
            if model.isOk:
                models.append(model)
                unique_blocks.add(blocks)  # Add blocks to the set
                correct_models += 1
                print(f"Successfully Created Model: {model.encoding}")
            else:
                print(f"Model creation failed for blocks: {blocks}")

        except Exception:
            # Same error message for all errors
            print("An error occurred while creating the model.")

    return models
