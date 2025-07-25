def stratified_sample_no_oversampling(
    X, y, z,
    group_fn,
    max_per_combo=250,
    min_per_combo=10
):
    from collections import defaultdict, Counter

    combo_to_indices = defaultdict(list)
    for i, (label, meta) in enumerate(zip(y, z)):
        skin_group = group_fn(meta.get("MST", -1))
        if skin_group != "unknown":
            combo_to_indices[(label, skin_group)].append(i)

    # Keep only combos with enough samples
    valid_combos = {k: v for k, v in combo_to_indices.items() if len(v) >= min_per_combo}

    sampled_indices = []
    sampled_combo_counts = Counter()

    for (cls, group), indices in valid_combos.items():
        selected = indices[:max_per_combo]  # no upsampling
        sampled_indices.extend(selected)
        sampled_combo_counts[(cls, group)] = len(selected)

    skipped_combos = sorted(set(combo_to_indices.keys()) - set(sampled_combo_counts.keys()))

    X_filtered = [X[i] for i in sampled_indices]
    y_filtered = [y[i] for i in sampled_indices]
    z_filtered = [z[i] for i in sampled_indices]

    print(f"✅ Sampled {len(sampled_indices)} total from {len(sampled_combo_counts)} combos")
    if skipped_combos:
        print(f"⚠️ Skipped {len(skipped_combos)} combos due to min_per_combo={min_per_combo}: {skipped_combos}")

    return X_filtered, y_filtered, z_filtered, sampled_combo_counts, skipped_combos