import random

def assignFitness(models):

    for model in models:
        # Assign FLOPs as a random value
        model.flops = random.uniform(1e6, 1e9)  # FLOPs range between 1M and 1B

        # Base PSNR correlates positively with FLOPs
        base_psnr = 20 + (model.flops / 1e9) * 20  # Base PSNR increases with FLOPs

        # Add small randomness to PSNR
        model.psnr = base_psnr + random.uniform(-3, 3)

        # Ensure PSNR stays within realistic bounds (20 to 40)
        model.psnr = max(20, min(model.psnr, 40))

    # Find min and max values for normalization
    min_flops = min(model.flops for model in models)
    max_flops = max(model.flops for model in models)
    min_psnr = min(model.psnr for model in models)
    max_psnr = max(model.psnr for model in models)

    # Normalize and calculate fitness score
    for model in models:
        # Normalize FLOPs
        if max_flops != min_flops:
            model.normalized_flops = (model.flops - min_flops) / (max_flops - min_flops)
        else:
            model.normalized_flops = 0

        # Normalize PSNR
        if max_psnr != min_psnr:
            model.normalized_psnr = (model.psnr - min_psnr) / (max_psnr - min_psnr)
        else:
            model.normalized_psnr = 0

        # Fitness score prioritizes PSNR but penalizes higher FLOPs
        model.fitness_score = model.normalized_psnr - 0.5 * model.normalized_flops

    return models
