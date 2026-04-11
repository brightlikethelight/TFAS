"""Canonical assembly of the s1s2 cognitive-bias benchmark.

This module owns the *content* of the benchmark: it defines the full
list of generator specs (one per pair) that, together, satisfy the
target counts in :data:`s1s2.benchmark.validate.TARGET_COUNTS`. The
specs themselves are explicit Python literals so that adding,
removing or editing one item is a single-file diff with version
history attached.

Why is this not a YAML file? Two reasons. First, the syllogism
specs need actual Python conditionals to validate the (valid xor
believable) constraint and to swap premises symbolically; YAML would
have to ship the same logic as code anyway. Second, keeping the
content next to the assembly logic makes it easy to enforce
"matching difficulty" and "no two pairs sharing a pair_id" with
``assert`` statements at construction time.

Design principles:

* All cover stories are NOVEL. We deliberately avoid the canonical
  exemplars (bat & ball, Linda, Asian disease, lily pad) so the
  benchmark stays as contamination-free as we can manage. The classic
  surface forms only appear inside ``provenance_note``.
* Every spec carries an integer ``difficulty`` in [1,5] and the
  conflict and control items in a pair always share that value (the
  generators enforce this).
* The number of specs per category equals the corresponding
  ``TARGET_COUNTS`` requirement so the validator passes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from beartype import beartype

from s1s2.benchmark.generators import (
    anchoring_isomorph,
    arithmetic_trap_isomorph,
    base_rate_isomorph,
    belief_bias_syllogism,
    conjunction_fallacy_isomorph,
    framing_isomorph,
    make_many,
)
from s1s2.benchmark.loader import BenchmarkItem
from s1s2.benchmark.templates import (
    bat_ball_isomorph,
    lily_pad_isomorph,
    widgets_machines_isomorph,
)

# --------------------------------------------------------------------- #
# CRT specs                                                             #
# --------------------------------------------------------------------- #


# 10 ratio (bat-and-ball-style) pairs.
_CRT_RATIO_SPECS: list[dict[str, Any]] = [
    {
        "pair_id": "crt_ratio_coffee_pastry",
        "object_a": "double espresso",
        "object_b": "biscotti",
        "total_cents": 660,
        "diff_cents": 600,
        "difficulty": 1,
    },
    {
        "pair_id": "crt_ratio_brush_paint",
        "object_a": "wide brush",
        "object_b": "tube of paint",
        "total_cents": 870,
        "diff_cents": 750,
        "difficulty": 1,
    },
    {
        "pair_id": "crt_ratio_helmet_lock",
        "object_a": "bicycle helmet",
        "object_b": "u-lock",
        "total_cents": 4500,
        "diff_cents": 4200,
        "difficulty": 2,
    },
    {
        "pair_id": "crt_ratio_kettle_strainer",
        "object_a": "copper kettle",
        "object_b": "tea strainer",
        "total_cents": 5510,
        "diff_cents": 5300,
        "difficulty": 3,
    },
    {
        "pair_id": "crt_ratio_compass_map",
        "object_a": "brass compass",
        "object_b": "trail map",
        "total_cents": 1320,
        "diff_cents": 1100,
        "difficulty": 3,
    },
    {
        "pair_id": "crt_ratio_pillow_case",
        "object_a": "down pillow",
        "object_b": "linen case",
        "total_cents": 4040,
        "diff_cents": 3700,
        "difficulty": 2,
    },
    {
        "pair_id": "crt_ratio_torch_battery",
        "object_a": "headlamp",
        "object_b": "spare battery",
        "total_cents": 2230,
        "diff_cents": 2050,
        "difficulty": 2,
    },
    {
        "pair_id": "crt_ratio_clay_glaze",
        "object_a": "block of clay",
        "object_b": "tin of glaze",
        "total_cents": 1810,
        "diff_cents": 1650,
        "difficulty": 3,
    },
    {
        "pair_id": "crt_ratio_ledger_pen",
        "object_a": "leather ledger",
        "object_b": "fountain pen",
        "total_cents": 9230,
        "diff_cents": 8900,
        "difficulty": 3,
    },
    {
        "pair_id": "crt_ratio_kite_string",
        "object_a": "delta kite",
        "object_b": "spool of string",
        "total_cents": 1450,
        "diff_cents": 1300,
        "difficulty": 1,
    },
]

# 10 work-rate pairs.
_CRT_WORK_RATE_SPECS: list[dict[str, Any]] = [
    {
        "pair_id": "crt_workrate_press_tile",
        "worker_label": "tile press",
        "output_label": "tile",
        "base_rate": 7,
        "scale": 84,
        "difficulty": 4,
    },
    {
        "pair_id": "crt_workrate_loom_scarf",
        "worker_label": "loom",
        "output_label": "scarf",
        "base_rate": 4,
        "scale": 48,
        "difficulty": 3,
    },
    {
        "pair_id": "crt_workrate_oven_loaf",
        "worker_label": "oven",
        "output_label": "loaf",
        "base_rate": 6,
        "scale": 90,
        "difficulty": 3,
    },
    {
        "pair_id": "crt_workrate_kiln_brick",
        "worker_label": "kiln",
        "output_label": "brick",
        "base_rate": 8,
        "scale": 96,
        "difficulty": 3,
    },
    {
        "pair_id": "crt_workrate_lathe_spindle",
        "worker_label": "lathe",
        "output_label": "spindle",
        "base_rate": 5,
        "scale": 75,
        "difficulty": 3,
    },
    {
        "pair_id": "crt_workrate_printer_label",
        "worker_label": "printer",
        "output_label": "label",
        "base_rate": 9,
        "scale": 108,
        "difficulty": 4,
    },
    {
        "pair_id": "crt_workrate_baler_bale",
        "worker_label": "baler",
        "output_label": "bale",
        "base_rate": 6,
        "scale": 72,
        "difficulty": 3,
    },
    {
        "pair_id": "crt_workrate_punch_button",
        "worker_label": "punch press",
        "output_label": "button",
        "base_rate": 10,
        "scale": 120,
        "difficulty": 3,
    },
    {
        "pair_id": "crt_workrate_filler_jar",
        "worker_label": "jar filler",
        "output_label": "jar",
        "base_rate": 7,
        "scale": 105,
        "difficulty": 4,
    },
    {
        "pair_id": "crt_workrate_packer_box",
        "worker_label": "packer",
        "output_label": "box",
        "base_rate": 12,
        "scale": 144,
        "difficulty": 4,
    },
]

# 10 exponential-doubling pairs.
_CRT_EXPGROWTH_SPECS: list[dict[str, Any]] = [
    {
        "pair_id": "crt_exp_moss_courtyard",
        "entity": "moss patches",
        "habitat": "stone courtyard",
        "days_to_full": 30,
        "difficulty": 3,
    },
    {
        "pair_id": "crt_exp_lichen_boulder",
        "entity": "lichen colony",
        "habitat": "boulder face",
        "days_to_full": 22,
        "difficulty": 3,
    },
    {
        "pair_id": "crt_exp_algae_tank",
        "entity": "algae",
        "habitat": "fish tank",
        "days_to_full": 14,
        "difficulty": 3,
    },
    {
        "pair_id": "crt_exp_bacteria_dish",
        "entity": "bacterial colony",
        "habitat": "petri dish",
        "days_to_full": 18,
        "difficulty": 3,
    },
    {
        "pair_id": "crt_exp_ivy_wall",
        "entity": "ivy",
        "habitat": "garden wall",
        "days_to_full": 26,
        "difficulty": 3,
    },
    {
        "pair_id": "crt_exp_mussels_pier",
        "entity": "freshwater mussels",
        "habitat": "pier piling",
        "days_to_full": 32,
        "difficulty": 4,
    },
    {
        "pair_id": "crt_exp_mold_loaf",
        "entity": "mold spores",
        "habitat": "bread loaf",
        "days_to_full": 12,
        "difficulty": 2,
    },
    {
        "pair_id": "crt_exp_termites_log",
        "entity": "termite swarm",
        "habitat": "fallen log",
        "days_to_full": 40,
        "difficulty": 4,
    },
    {
        "pair_id": "crt_exp_kelp_pen",
        "entity": "kelp",
        "habitat": "saltwater pen",
        "days_to_full": 16,
        "difficulty": 3,
    },
    {
        "pair_id": "crt_exp_phytoplankton_pond",
        "entity": "phytoplankton",
        "habitat": "shallow pond",
        "days_to_full": 20,
        "difficulty": 4,
    },
]


# --------------------------------------------------------------------- #
# base-rate specs                                                       #
# --------------------------------------------------------------------- #


# 20 pairs.
_BASE_RATE_SPECS: list[dict[str, Any]] = [
    {
        "pair_id": "br_brewer_developer",
        "description": (
            "She wakes early, brews her own kombucha, keeps a "
            "sourdough starter on the counter, and writes a blog "
            "about fermented vegetables."
        ),
        "rare_group": "professional brewmaster",
        "common_group": "software developer",
        "rare_rate": 0.01,
        "common_rate": 0.65,
        "difficulty": 2,
    },
    {
        "pair_id": "br_falconer_accountant",
        "description": (
            "On weekends he hikes ridgelines with binoculars, keeps "
            "raptor identification charts in his glovebox, and has "
            "rebuilt the leather hood collection on a wall in his garage."
        ),
        "rare_group": "professional falconer",
        "common_group": "accountant",
        "rare_rate": 0.005,
        "common_rate": 0.55,
        "difficulty": 2,
    },
    {
        "pair_id": "br_glassblower_nurse",
        "description": (
            "Her hands are steady, she enjoys precise temperature "
            "control on her gas stove, and she has a small kiln in "
            "the corner of her studio apartment."
        ),
        "rare_group": "professional glassblower",
        "common_group": "registered nurse",
        "rare_rate": 0.01,
        "common_rate": 0.50,
        "difficulty": 3,
    },
    {
        "pair_id": "br_arctic_geologist_teacher",
        "description": (
            "He owns three different pairs of insulated bib overalls, "
            "spends his vacations in subarctic latitudes, and "
            "subscribes to a quarterly journal on permafrost."
        ),
        "rare_group": "arctic field geologist",
        "common_group": "elementary school teacher",
        "rare_rate": 0.002,
        "common_rate": 0.45,
        "difficulty": 3,
    },
    {
        "pair_id": "br_calligrapher_clerk",
        "description": (
            "She owns a collection of sumi inks, hand-letters her "
            "wedding invitations, and once won a regional brush-script "
            "competition in her teens."
        ),
        "rare_group": "professional calligrapher",
        "common_group": "office clerk",
        "rare_rate": 0.005,
        "common_rate": 0.40,
        "difficulty": 2,
    },
    {
        "pair_id": "br_taxidermist_warehouse",
        "description": (
            "He keeps detailed wildlife sketches, owns a freezer "
            "dedicated to specimens, and reads anatomy textbooks for fun."
        ),
        "rare_group": "professional taxidermist",
        "common_group": "warehouse worker",
        "rare_rate": 0.001,
        "common_rate": 0.50,
        "difficulty": 3,
    },
    {
        "pair_id": "br_violinmaker_driver",
        "description": (
            "He has callused fingertips, owns three sets of luthier's "
            "chisels, and his apartment smells faintly of varnish."
        ),
        "rare_group": "violin maker",
        "common_group": "delivery driver",
        "rare_rate": 0.002,
        "common_rate": 0.45,
        "difficulty": 3,
    },
    {
        "pair_id": "br_perfumer_salesclerk",
        "description": (
            "She has an unusually sensitive nose, owns a hundred small "
            "vials of essential oils, and enjoys naming the precise "
            "components of any room she enters."
        ),
        "rare_group": "professional perfumer",
        "common_group": "retail sales clerk",
        "rare_rate": 0.003,
        "common_rate": 0.55,
        "difficulty": 2,
    },
    {
        "pair_id": "br_arborist_admin",
        "description": (
            "He owns climbing harnesses, has biceps thick from years "
            "of pruning, and his car has a permanent layer of bark dust."
        ),
        "rare_group": "professional arborist",
        "common_group": "administrative assistant",
        "rare_rate": 0.005,
        "common_rate": 0.50,
        "difficulty": 2,
    },
    {
        "pair_id": "br_bookbinder_cashier",
        "description": (
            "Her shelves are crammed with hand-stitched journals, she "
            "collects bone folders, and her thumb has a permanent "
            "indentation from years of folding signatures."
        ),
        "rare_group": "professional bookbinder",
        "common_group": "cashier",
        "rare_rate": 0.002,
        "common_rate": 0.50,
        "difficulty": 3,
    },
    {
        "pair_id": "br_volcanologist_engineer",
        "description": (
            "He owns multiple high-temperature gloves, his vacation "
            "photos always feature lava, and his bookshelf is full of "
            "geophysics textbooks."
        ),
        "rare_group": "field volcanologist",
        "common_group": "civil engineer",
        "rare_rate": 0.001,
        "common_rate": 0.45,
        "difficulty": 3,
    },
    {
        "pair_id": "br_dressage_clerk",
        "description": (
            "She owns three sets of riding boots, follows international "
            "equestrian tournaments closely, and her Instagram is full "
            "of horses."
        ),
        "rare_group": "professional dressage rider",
        "common_group": "bank clerk",
        "rare_rate": 0.0005,
        "common_rate": 0.55,
        "difficulty": 2,
    },
    {
        "pair_id": "br_acrobat_logistics",
        "description": (
            "He stretches every morning, can hold a one-handed "
            "handstand for thirty seconds, and once spent a year "
            "training in Montreal."
        ),
        "rare_group": "circus acrobat",
        "common_group": "logistics coordinator",
        "rare_rate": 0.0008,
        "common_rate": 0.45,
        "difficulty": 3,
    },
    {
        "pair_id": "br_blacksmith_office",
        "description": (
            "He has burn scars on his forearms, his garage contains "
            "an anvil and a propane forge, and he hammers spoons out "
            "of iron rod for fun."
        ),
        "rare_group": "professional blacksmith",
        "common_group": "office worker",
        "rare_rate": 0.001,
        "common_rate": 0.55,
        "difficulty": 2,
    },
    {
        "pair_id": "br_archeologist_clerk",
        "description": (
            "She has a fondness for Bronze Age pottery, owns a set of "
            "trowels and brushes, and her summer holidays are always "
            "spent at field digs."
        ),
        "rare_group": "field archeologist",
        "common_group": "insurance clerk",
        "rare_rate": 0.002,
        "common_rate": 0.50,
        "difficulty": 3,
    },
    {
        "pair_id": "br_marine_biologist_secretary",
        "description": (
            "She has a saltwater aquarium, dives recreationally, and "
            "subscribes to two journals on coral reef ecology."
        ),
        "rare_group": "marine biologist",
        "common_group": "secretary",
        "rare_rate": 0.003,
        "common_rate": 0.50,
        "difficulty": 2,
    },
    {
        "pair_id": "br_smokejumper_admin",
        "description": (
            "He keeps a packed bag in his hallway, is in extreme "
            "physical condition, and his vacation photos always show "
            "burned forest."
        ),
        "rare_group": "smokejumper",
        "common_group": "administrative manager",
        "rare_rate": 0.0005,
        "common_rate": 0.50,
        "difficulty": 5,
    },
    {
        "pair_id": "br_horologist_cashier",
        "description": (
            "He owns three loupes, his desk is covered in tweezers "
            "and tiny screwdrivers, and he can identify a watch "
            "movement by sound alone."
        ),
        "rare_group": "professional horologist",
        "common_group": "supermarket cashier",
        "rare_rate": 0.0008,
        "common_rate": 0.55,
        "difficulty": 3,
    },
    {
        "pair_id": "br_mahout_clerk",
        "description": (
            "He has spent a decade working with very large mammals "
            "in southeast Asia, knows roughly fifty commands in Khmer, "
            "and his social media is mostly elephant photos."
        ),
        "rare_group": "elephant mahout",
        "common_group": "post office clerk",
        "rare_rate": 0.0001,
        "common_rate": 0.45,
        "difficulty": 5,
    },
    {
        "pair_id": "br_lapidary_factory",
        "description": (
            "He owns three rotary saws sized for stone, has a "
            "garage full of agate and jasper rough, and gives polished "
            "thumb stones as gifts."
        ),
        "rare_group": "professional lapidary",
        "common_group": "factory worker",
        "rare_rate": 0.001,
        "common_rate": 0.55,
        "difficulty": 3,
    },
]


# --------------------------------------------------------------------- #
# belief-bias syllogism specs                                           #
# --------------------------------------------------------------------- #


# 25 pairs. Each pair includes BOTH cells of a 2x2 belief-bias
# stimulus: the conflict cell (validity XOR believability) and the
# matched control cell (the opposite-validity counterpart on the same
# believability axis).
#
# Why explicit control premises? See `belief_bias_syllogism` --
# auto-flipping a syllogism's major premise does not reliably convert
# valid into invalid across all four moods, so we trust the human
# author per spec instead.
#
# Logical templates used here:
#   - VALID Barbara (AAA-1): "All M are P; All S are M; therefore
#     All S are P." Used for the (valid, unbelievable) conflict cell
#     and its (invalid, unbelievable) control by reordering to AAA-2
#     ("All P are M; All S are M") which is undistributed middle.
#   - INVALID undistributed middle (AAA-2): "All P are M; All S are
#     M; therefore All S are P." Used for the (invalid, believable)
#     conflict and paired with a valid Barbara as control.
#   - INVALID illicit minor (AAA-3): "All M are P; All M are S;
#     therefore All S are P." Distinct invalid form for variety.
#
# We balance across the two conflict cells: 12 (valid, unbelievable)
# + 13 (invalid, believable) = 25.
_SYLLOGISM_SPECS: list[dict[str, Any]] = [
    # ---- (valid, unbelievable) conflict cells: AAA-1 with an
    # implausible major; the matched control is AAA-2 with the SAME
    # implausible major's content but reordered into an invalid form.
    {
        "pair_id": "syll_swimmers_landlocked",
        # VALID Barbara: All M(swimmers) are P(landlocked-born); All
        # S(team) are M; therefore All S are P.
        "conflict_major": "All Olympic swimmers were born in landlocked countries.",
        "conflict_minor": "All members of the Italian relay team are Olympic swimmers.",
        "conflict_conclusion": (
            "All members of the Italian relay team were born in "
            "landlocked countries."
        ),
        "conflict_is_valid": True,
        "is_believable": False,
        # INVALID AAA-2: "All P are M; All S are M; therefore All S
        # are P." (undistributed middle)
        "control_major": "All people born in landlocked countries are Olympic swimmers.",
        "control_minor": "All members of the Italian relay team are Olympic swimmers.",
        "difficulty": 4,
    },
    {
        "pair_id": "syll_mammal_breath",
        "conflict_major": "All mammals can hold their breath underwater for at least one hour.",
        "conflict_minor": "All hippos are mammals.",
        "conflict_conclusion": "All hippos can hold their breath underwater for at least one hour.",
        "conflict_is_valid": True,
        "is_believable": False,
        "control_major": "All animals that can hold their breath underwater for at least one hour are mammals.",
        "control_minor": "All hippos can hold their breath underwater for at least one hour.",
        "difficulty": 3,
    },
    {
        "pair_id": "syll_metal_corrodes",
        "conflict_major": "All gold objects rust away to nothing within a single year.",
        "conflict_minor": "All wedding rings in the museum vault are gold objects.",
        "conflict_conclusion": "All wedding rings in the museum vault rust away to nothing within a single year.",
        "conflict_is_valid": True,
        "is_believable": False,
        "control_major": "All things that rust away to nothing within a single year are gold objects.",
        "control_minor": "All wedding rings in the museum vault rust away to nothing within a single year.",
        "difficulty": 4,
    },
    {
        "pair_id": "syll_evergreen_lose_leaves",
        "conflict_major": "All conifers shed every needle each winter.",
        "conflict_minor": "All Scots pines are conifers.",
        "conflict_conclusion": "All Scots pines shed every needle each winter.",
        "conflict_is_valid": True,
        "is_believable": False,
        "control_major": "All trees that shed every needle each winter are conifers.",
        "control_minor": "All Scots pines shed every needle each winter.",
        "difficulty": 2,
    },
    {
        "pair_id": "syll_predator_vegetarian",
        "conflict_major": "All eagles are strict vegetarians who never eat meat.",
        "conflict_minor": "All bald eagles are eagles.",
        "conflict_conclusion": "All bald eagles are strict vegetarians who never eat meat.",
        "conflict_is_valid": True,
        "is_believable": False,
        "control_major": "All strict vegetarians who never eat meat are eagles.",
        "control_minor": "All bald eagles are strict vegetarians who never eat meat.",
        "difficulty": 2,
    },
    {
        "pair_id": "syll_deserts_cold",
        "conflict_major": "All deserts remain below freezing year-round.",
        "conflict_minor": "All locations in the central Sahara are deserts.",
        "conflict_conclusion": "All locations in the central Sahara remain below freezing year-round.",
        "conflict_is_valid": True,
        "is_believable": False,
        "control_major": "All locations that remain below freezing year-round are deserts.",
        "control_minor": "All locations in the central Sahara remain below freezing year-round.",
        "difficulty": 3,
    },
    {
        "pair_id": "syll_cats_aquatic",
        "conflict_major": "All housecats are aquatic animals that live their entire lives underwater.",
        "conflict_minor": "All Siamese cats are housecats.",
        "conflict_conclusion": "All Siamese cats are aquatic animals that live their entire lives underwater.",
        "conflict_is_valid": True,
        "is_believable": False,
        "control_major": "All aquatic animals that live their entire lives underwater are housecats.",
        "control_minor": "All Siamese cats are aquatic animals that live their entire lives underwater.",
        "difficulty": 2,
    },
    {
        "pair_id": "syll_glass_dense",
        "conflict_major": "All glass spheres are denser than lead.",
        "conflict_minor": "All marbles in the toy shop are glass spheres.",
        "conflict_conclusion": "All marbles in the toy shop are denser than lead.",
        "conflict_is_valid": True,
        "is_believable": False,
        "control_major": "All objects denser than lead are glass spheres.",
        "control_minor": "All marbles in the toy shop are denser than lead.",
        "difficulty": 2,
    },
    {
        "pair_id": "syll_owl_diurnal",
        "conflict_major": "All owls hunt only during full daylight.",
        "conflict_minor": "All barn owls are owls.",
        "conflict_conclusion": "All barn owls hunt only during full daylight.",
        "conflict_is_valid": True,
        "is_believable": False,
        "control_major": "All birds that hunt only during full daylight are owls.",
        "control_minor": "All barn owls hunt only during full daylight.",
        "difficulty": 2,
    },
    {
        "pair_id": "syll_water_combustible",
        "conflict_major": "All foods that contain water are highly explosive when ignited.",
        "conflict_minor": "All cucumbers are foods that contain water.",
        "conflict_conclusion": "All cucumbers are highly explosive when ignited.",
        "conflict_is_valid": True,
        "is_believable": False,
        "control_major": "All things highly explosive when ignited are foods that contain water.",
        "control_minor": "All cucumbers are highly explosive when ignited.",
        "difficulty": 3,
    },
    {
        "pair_id": "syll_planets_cubes",
        "conflict_major": "All planets are perfect cubes with sharp corners.",
        "conflict_minor": "All inner-system planets are planets.",
        "conflict_conclusion": "All inner-system planets are perfect cubes with sharp corners.",
        "conflict_is_valid": True,
        "is_believable": False,
        "control_major": "All perfect cubes with sharp corners are planets.",
        "control_minor": "All inner-system planets are perfect cubes with sharp corners.",
        "difficulty": 2,
    },
    {
        "pair_id": "syll_philosophers_immortal",
        "conflict_major": "All philosophers are immortal beings who have lived for thousands of years.",
        "conflict_minor": "All ancient Greek philosophers are philosophers.",
        "conflict_conclusion": "All ancient Greek philosophers are immortal beings who have lived for thousands of years.",
        "conflict_is_valid": True,
        "is_believable": False,
        "control_major": "All immortal beings who have lived for thousands of years are philosophers.",
        "control_minor": "All ancient Greek philosophers are immortal beings who have lived for thousands of years.",
        "difficulty": 3,
    },
    # ---- (invalid, believable) conflict cells: AAA-2 (undistributed
    # middle) with a plausible-sounding conclusion. Matched control
    # is the same content rearranged into VALID Barbara.
    {
        "pair_id": "syll_dogs_pets",
        # INVALID AAA-2: All P are M; All S are M; therefore All S
        # are P. Concretely: All pets are beagles; All dogs are
        # beagles; therefore all dogs are pets. The conclusion
        # sounds plausible.
        "conflict_major": "All pets are beagles.",
        "conflict_minor": "All dogs are beagles.",
        "conflict_conclusion": "All dogs are pets.",
        "conflict_is_valid": False,
        "is_believable": True,
        # VALID Barbara restoration of the same surface topic.
        "control_major": "All beagles are pets.",
        "control_minor": "All dogs are beagles.",
        "difficulty": 3,
    },
    {
        "pair_id": "syll_doctors_educated",
        "conflict_major": "All highly educated people are cardiologists.",
        "conflict_minor": "All doctors are cardiologists.",
        "conflict_conclusion": "All doctors are highly educated people.",
        "conflict_is_valid": False,
        "is_believable": True,
        "control_major": "All cardiologists are highly educated people.",
        "control_minor": "All doctors are cardiologists.",
        "difficulty": 3,
    },
    {
        "pair_id": "syll_apples_grow",
        "conflict_major": "All things that grow on trees are Honeycrisp apples.",
        "conflict_minor": "All fruits are Honeycrisp apples.",
        "conflict_conclusion": "All fruits grow on trees.",
        "conflict_is_valid": False,
        "is_believable": True,
        "control_major": "All Honeycrisp apples grow on trees.",
        "control_minor": "All fruits are Honeycrisp apples.",
        "difficulty": 3,
    },
    {
        "pair_id": "syll_athletes_fit",
        "conflict_major": "All physically fit individuals are marathon runners.",
        "conflict_minor": "All athletes are marathon runners.",
        "conflict_conclusion": "All athletes are physically fit individuals.",
        "conflict_is_valid": False,
        "is_believable": True,
        "control_major": "All marathon runners are physically fit individuals.",
        "control_minor": "All athletes are marathon runners.",
        "difficulty": 3,
    },
    {
        "pair_id": "syll_metals_conduct",
        "conflict_major": "All things that conduct electricity are copper wires.",
        "conflict_minor": "All metals are copper wires.",
        "conflict_conclusion": "All metals conduct electricity.",
        "conflict_is_valid": False,
        "is_believable": True,
        "control_major": "All copper wires conduct electricity.",
        "control_minor": "All metals are copper wires.",
        "difficulty": 5,
    },
    {
        "pair_id": "syll_cars_wheels",
        "conflict_major": "All vehicles with four wheels are sedans.",
        "conflict_minor": "All cars are sedans.",
        "conflict_conclusion": "All cars are vehicles with four wheels.",
        "conflict_is_valid": False,
        "is_believable": True,
        "control_major": "All sedans are vehicles with four wheels.",
        "control_minor": "All cars are sedans.",
        "difficulty": 3,
    },
    {
        "pair_id": "syll_birds_fly",
        "conflict_major": "All things that can fly are robins.",
        "conflict_minor": "All birds are robins.",
        "conflict_conclusion": "All birds can fly.",
        "conflict_is_valid": False,
        "is_believable": True,
        "control_major": "All robins can fly.",
        "control_minor": "All birds are robins.",
        "difficulty": 3,
    },
    {
        "pair_id": "syll_fish_swim",
        "conflict_major": "All animals that swim are trout.",
        "conflict_minor": "All fish are trout.",
        "conflict_conclusion": "All fish swim.",
        "conflict_is_valid": False,
        "is_believable": True,
        "control_major": "All trout swim.",
        "control_minor": "All fish are trout.",
        "difficulty": 3,
    },
    {
        "pair_id": "syll_tools_metal",
        "conflict_major": "All things made of metal are wrenches.",
        "conflict_minor": "All tools are wrenches.",
        "conflict_conclusion": "All tools are made of metal.",
        "conflict_is_valid": False,
        "is_believable": True,
        "control_major": "All wrenches are made of metal.",
        "control_minor": "All tools are wrenches.",
        "difficulty": 3,
    },
    {
        "pair_id": "syll_books_paper",
        "conflict_major": "All things made of paper are hardcover novels.",
        "conflict_minor": "All books are hardcover novels.",
        "conflict_conclusion": "All books are made of paper.",
        "conflict_is_valid": False,
        "is_believable": True,
        "control_major": "All hardcover novels are made of paper.",
        "control_minor": "All books are hardcover novels.",
        "difficulty": 3,
    },
    {
        "pair_id": "syll_buildings_stand",
        "conflict_major": "All things that stand on bedrock are skyscrapers in Manhattan.",
        "conflict_minor": "All buildings are skyscrapers in Manhattan.",
        "conflict_conclusion": "All buildings stand on bedrock.",
        "conflict_is_valid": False,
        "is_believable": True,
        "control_major": "All skyscrapers in Manhattan stand on bedrock.",
        "control_minor": "All buildings are skyscrapers in Manhattan.",
        "difficulty": 5,
    },
    {
        "pair_id": "syll_chairs_legs",
        "conflict_major": "All things with four legs are wooden dining chairs.",
        "conflict_minor": "All chairs are wooden dining chairs.",
        "conflict_conclusion": "All chairs have four legs.",
        "conflict_is_valid": False,
        "is_believable": True,
        "control_major": "All wooden dining chairs have four legs.",
        "control_minor": "All chairs are wooden dining chairs.",
        "difficulty": 3,
    },
    {
        "pair_id": "syll_houses_doors",
        "conflict_major": "All buildings with front doors are colonial cottages.",
        "conflict_minor": "All houses are colonial cottages.",
        "conflict_conclusion": "All houses have front doors.",
        "conflict_is_valid": False,
        "is_believable": True,
        "control_major": "All colonial cottages have front doors.",
        "control_minor": "All houses are colonial cottages.",
        "difficulty": 3,
    },
]


# --------------------------------------------------------------------- #
# anchoring specs                                                       #
# --------------------------------------------------------------------- #


# 15 pairs. Pick fact-based questions where the true value is well-
# established and the high anchor is clearly above it.
_ANCHORING_SPECS: list[dict[str, Any]] = [
    {
        "pair_id": "anchor_un_members",
        "question": (
            "How many UN member states were there as of 2023?"
        ),
        "true_value": 193,
        "high_anchor": 350,
        "low_anchor": 50,
        "units": "member states",
        "difficulty": 1,
    },
    {
        "pair_id": "anchor_periodic_elements",
        "question": (
            "How many chemical elements appear on the standard "
            "periodic table as of 2023?"
        ),
        "true_value": 118,
        "high_anchor": 250,
        "low_anchor": 40,
        "units": "elements",
        "difficulty": 1,
    },
    {
        "pair_id": "anchor_country_capital_year",
        "question": (
            "In what year was the Australian capital city of Canberra "
            "officially founded?"
        ),
        "true_value": 1913,
        "high_anchor": 1980,
        "low_anchor": 1750,
        "units": "AD",
        "difficulty": 3,
    },
    {
        "pair_id": "anchor_amazon_river_length",
        "question": (
            "Approximately how long is the Amazon River, in kilometres?"
        ),
        "true_value": 6400,
        "high_anchor": 14000,
        "low_anchor": 1000,
        "units": "kilometres",
        "difficulty": 3,
    },
    {
        "pair_id": "anchor_human_chromosomes",
        "question": (
            "How many chromosomes are in a typical human somatic cell?"
        ),
        "true_value": 46,
        "high_anchor": 200,
        "low_anchor": 8,
        "units": "chromosomes",
        "difficulty": 1,
    },
    {
        "pair_id": "anchor_sun_radius",
        "question": (
            "Approximately how many times larger is the radius of "
            "the Sun than the radius of Earth?"
        ),
        "true_value": 109,
        "high_anchor": 500,
        "low_anchor": 10,
        "units": "times",
        "difficulty": 3,
    },
    {
        "pair_id": "anchor_hist_olympics",
        "question": (
            "In what year were the first modern Olympic Games held in "
            "Athens?"
        ),
        "true_value": 1896,
        "high_anchor": 1960,
        "low_anchor": 1700,
        "units": "AD",
        "difficulty": 1,
    },
    {
        "pair_id": "anchor_keys_piano",
        "question": (
            "How many keys does a standard modern grand piano have?"
        ),
        "true_value": 88,
        "high_anchor": 200,
        "low_anchor": 30,
        "units": "keys",
        "difficulty": 1,
    },
    {
        "pair_id": "anchor_great_wall",
        "question": (
            "Approximately how long is the Great Wall of China, in "
            "kilometres, including all branches?"
        ),
        "true_value": 21000,
        "high_anchor": 60000,
        "low_anchor": 3000,
        "units": "kilometres",
        "difficulty": 3,
    },
    {
        "pair_id": "anchor_marathon_distance",
        "question": (
            "What is the official distance of an Olympic marathon, "
            "in metres?"
        ),
        "true_value": 42195,
        "high_anchor": 90000,
        "low_anchor": 10000,
        "units": "metres",
        "difficulty": 1,
    },
    {
        "pair_id": "anchor_mount_fuji",
        "question": (
            "Approximately how tall is Mount Fuji, in metres above "
            "sea level?"
        ),
        "true_value": 3776,
        "high_anchor": 8000,
        "low_anchor": 800,
        "units": "metres",
        "difficulty": 2,
    },
    {
        "pair_id": "anchor_dna_pairs",
        "question": (
            "Approximately how many base pairs are in the haploid "
            "human genome, in millions?"
        ),
        "true_value": 3000,
        "high_anchor": 12000,
        "low_anchor": 100,
        "units": "million base pairs",
        "difficulty": 5,
    },
    {
        "pair_id": "anchor_us_states",
        "question": (
            "How many states are in the United States as of 2023?"
        ),
        "true_value": 50,
        "high_anchor": 120,
        "low_anchor": 13,
        "units": "states",
        "difficulty": 1,
    },
    {
        "pair_id": "anchor_speed_light",
        "question": (
            "Approximately how fast does light travel in a vacuum, "
            "in thousand kilometres per second?"
        ),
        "true_value": 300,
        "high_anchor": 1500,
        "low_anchor": 50,
        "units": "thousand km/s",
        "difficulty": 3,
    },
    {
        "pair_id": "anchor_everest_meters",
        "question": (
            "Approximately how tall is Mount Everest, in metres above "
            "sea level?"
        ),
        "true_value": 8849,
        "high_anchor": 20000,
        "low_anchor": 1000,
        "units": "metres",
        "difficulty": 2,
    },
]


# --------------------------------------------------------------------- #
# framing specs                                                         #
# --------------------------------------------------------------------- #


# 15 pairs. We avoid 'Asian disease' surface form: use distinct
# scenarios. Numbers are picked so EVs match exactly.
def _frame(
    pair_id: str,
    scenario: str,
    n_total: int,
    n_certain_save: int,
    prob_save_all: float,
    prefer_sure: bool,
    difficulty: int = 2,
) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "scenario": scenario,
        "n_total": n_total,
        "n_certain_save": n_certain_save,
        "prob_save_all": prob_save_all,
        "prefer_sure": prefer_sure,
        "difficulty": difficulty,
    }


_FRAMING_SPECS: list[dict[str, Any]] = [
    _frame(
        "frame_flood_village",
        (
            "A river is rising in a small farming village in the "
            "valley below."
        ),
        n_total=600,
        n_certain_save=200,
        prob_save_all=1 / 3,
        prefer_sure=True,
        difficulty=2,
    ),
    _frame(
        "frame_wildfire_canyon",
        (
            "A wildfire is sweeping toward a row of houses in a "
            "canyon community."
        ),
        n_total=900,
        n_certain_save=300,
        prob_save_all=1 / 3,
        prefer_sure=True,
        difficulty=2,
    ),
    _frame(
        "frame_avalanche_resort",
        (
            "An avalanche is bearing down on a ski resort whose "
            "buildings are full of guests."
        ),
        n_total=400,
        n_certain_save=100,
        prob_save_all=0.25,
        prefer_sure=True,
        difficulty=3,
    ),
    _frame(
        "frame_chemical_spill",
        (
            "A chemical tanker has overturned upwind of a small town."
        ),
        n_total=1000,
        n_certain_save=500,
        prob_save_all=0.5,
        prefer_sure=False,
        difficulty=2,
    ),
    _frame(
        "frame_dam_failing",
        (
            "An aging dam is showing cracks above a small town."
        ),
        n_total=500,
        n_certain_save=100,
        prob_save_all=0.2,
        prefer_sure=False,
        difficulty=3,
    ),
    _frame(
        "frame_ferry_capsize",
        (
            "A passenger ferry has begun to take on water in cold "
            "northern seas."
        ),
        n_total=300,
        n_certain_save=75,
        prob_save_all=0.25,
        prefer_sure=True,
        difficulty=3,
    ),
    _frame(
        "frame_storm_islanders",
        (
            "A category-five storm is approaching a chain of small "
            "inhabited islands."
        ),
        n_total=800,
        n_certain_save=200,
        prob_save_all=0.25,
        prefer_sure=False,
        difficulty=3,
    ),
    _frame(
        "frame_landmine_field",
        (
            "A field used by villagers as a shortcut has just been "
            "discovered to be heavily mined."
        ),
        n_total=200,
        n_certain_save=50,
        prob_save_all=0.25,
        prefer_sure=True,
        difficulty=3,
    ),
    _frame(
        "frame_blizzard_pass",
        (
            "A blizzard has stranded climbers on a high mountain pass."
        ),
        n_total=120,
        n_certain_save=40,
        prob_save_all=1 / 3,
        prefer_sure=False,
        difficulty=3,
    ),
    _frame(
        "frame_carbon_monoxide",
        (
            "A carbon-monoxide leak has been detected in a large "
            "apartment building at night."
        ),
        n_total=240,
        n_certain_save=80,
        prob_save_all=1 / 3,
        prefer_sure=True,
        difficulty=2,
    ),
    _frame(
        "frame_volcano_evacuation",
        (
            "A volcano is showing signs of imminent eruption near a "
            "town in the foothills."
        ),
        n_total=1500,
        n_certain_save=300,
        prob_save_all=0.2,
        prefer_sure=False,
        difficulty=5,
    ),
    _frame(
        "frame_collapse_mine",
        (
            "A coal mine has partially collapsed with miners trapped "
            "in the lower galleries."
        ),
        n_total=80,
        n_certain_save=20,
        prob_save_all=0.25,
        prefer_sure=True,
        difficulty=3,
    ),
    _frame(
        "frame_marathon_heatstroke",
        (
            "A marathon is being held under a sudden heat wave with "
            "runners collapsing across the course."
        ),
        n_total=900,
        n_certain_save=300,
        prob_save_all=1 / 3,
        prefer_sure=False,
        difficulty=2,
    ),
    _frame(
        "frame_outbreak_camp",
        (
            "A respiratory outbreak has begun in a refugee camp in "
            "winter."
        ),
        n_total=600,
        n_certain_save=150,
        prob_save_all=0.25,
        prefer_sure=True,
        difficulty=3,
    ),
    _frame(
        "frame_plane_crash",
        (
            "A small commuter plane has gone down in remote forest "
            "with passengers needing rescue."
        ),
        n_total=40,
        n_certain_save=10,
        prob_save_all=0.25,
        prefer_sure=False,
        difficulty=3,
    ),
]


# --------------------------------------------------------------------- #
# conjunction-fallacy specs                                             #
# --------------------------------------------------------------------- #


# 12 pairs.
_CONJUNCTION_SPECS: list[dict[str, Any]] = [
    {
        "pair_id": "conj_marcus_climber",
        "person_name": "Marcus",
        "person_description": (
            "Marcus, age 34, has callused hands, a deep tan, and an "
            "apartment full of carabiners. He keeps a chalk bag clipped "
            "to his belt and his weekends are spent on remote granite "
            "walls."
        ),
        "feature_a": "graduate student",
        "feature_b": "competitive rock climber",
        "difficulty": 3,
    },
    {
        "pair_id": "conj_priya_violinist",
        "person_name": "Priya",
        "person_description": (
            "Priya, age 28, has a delicate touch and impeccable "
            "rhythm. She practices three hours a day on an instrument "
            "she has owned since age six and has performed at small "
            "concert halls across Europe."
        ),
        "feature_a": "office worker",
        "feature_b": "professional classical violinist",
        "difficulty": 3,
    },
    {
        "pair_id": "conj_elena_pacifist",
        "person_name": "Elena",
        "person_description": (
            "Elena, age 41, donates monthly to environmental groups, "
            "wears recycled clothing, and has tattoos of endangered "
            "species on both forearms. She speaks four languages and "
            "lives without a car."
        ),
        "feature_a": "civil engineer",
        "feature_b": "environmental activist who organizes protests",
        "difficulty": 3,
    },
    {
        "pair_id": "conj_omar_chef",
        "person_name": "Omar",
        "person_description": (
            "Omar, age 36, has burn scars across his forearms, owns "
            "an enormous knife collection, and his Instagram is full "
            "of plated food. He talks about reductions and emulsions "
            "the way other people talk about the weather."
        ),
        "feature_a": "small business owner",
        "feature_b": "professionally trained French chef",
        "difficulty": 2,
    },
    {
        "pair_id": "conj_sarah_anthropologist",
        "person_name": "Sarah",
        "person_description": (
            "Sarah, age 39, lived for two years in a remote highland "
            "village in Papua New Guinea. She speaks the local "
            "language fluently, owns ceremonial textiles she traded "
            "for in the field, and writes long ethnographic essays."
        ),
        "feature_a": "writer",
        "feature_b": "field anthropologist who studies highland societies",
        "difficulty": 3,
    },
    {
        "pair_id": "conj_diego_marathoner",
        "person_name": "Diego",
        "person_description": (
            "Diego, age 32, runs at 5am every day, eats a meticulously "
            "calibrated diet, and his living-room shelf holds half a "
            "dozen finishing medals from major events."
        ),
        "feature_a": "athlete",
        "feature_b": "competitive long-distance marathon runner",
        "difficulty": 2,
    },
    {
        "pair_id": "conj_maya_painter",
        "person_name": "Maya",
        "person_description": (
            "Maya, age 26, has paint-stained fingertips, owns a "
            "carefully organized rack of forty oil tubes, and her "
            "small studio apartment is dominated by an easel and "
            "stretched canvases in various states of completion."
        ),
        "feature_a": "graduate student",
        "feature_b": "regular gallery exhibitor",
        "difficulty": 2,
    },
    {
        "pair_id": "conj_yusuf_chess",
        "person_name": "Yusuf",
        "person_description": (
            "Yusuf, age 30, has an unusual ability to recall long "
            "sequences. He plays speed games online for several hours "
            "a day and his bookshelf is filled with annotated game "
            "collections by Soviet grandmasters."
        ),
        "feature_a": "consultant",
        "feature_b": "tournament-rated chess master",
        "difficulty": 3,
    },
    {
        "pair_id": "conj_lena_biologist",
        "person_name": "Lena",
        "person_description": (
            "Lena, age 33, has a microscope on her kitchen counter, "
            "keeps mosquito-rearing trays in her spare room, and "
            "spends her vacations collecting specimens in tropical "
            "wetlands."
        ),
        "feature_a": "scientist",
        "feature_b": "tropical-disease epidemiologist",
        "difficulty": 3,
    },
    {
        "pair_id": "conj_thomas_potter",
        "person_name": "Thomas",
        "person_description": (
            "Thomas, age 45, has dust permanently embedded in the "
            "creases of his hands, owns three kick wheels, and his "
            "garage is filled with shelves of bisque-fired bowls "
            "awaiting glazing."
        ),
        "feature_a": "self-employed worker",
        "feature_b": "professional ceramicist",
        "difficulty": 2,
    },
    {
        "pair_id": "conj_anna_birdwatcher",
        "person_name": "Anna",
        "person_description": (
            "Anna, age 52, owns three pairs of binoculars and a "
            "spotting scope, keeps a meticulous life list, and her "
            "vacations are organized around migration windows in "
            "various countries."
        ),
        "feature_a": "retired person",
        "feature_b": "experienced ornithologist",
        "difficulty": 2,
    },
    {
        "pair_id": "conj_kenji_cyclist",
        "person_name": "Kenji",
        "person_description": (
            "Kenji, age 29, has the lean musculature of an endurance "
            "athlete, owns four custom-built bicycles, and rides at "
            "least two-hundred kilometres every weekend on alpine "
            "passes near his home."
        ),
        "feature_a": "white-collar worker",
        "feature_b": "competitive amateur road cyclist",
        "difficulty": 3,
    },
]


# --------------------------------------------------------------------- #
# arithmetic-trap specs                                                 #
# --------------------------------------------------------------------- #


# 25 pairs. Each spec uses small integers so the lure is unambiguous.
def _ar(
    pair_id: str,
    start: int,
    steps: list[tuple[str, int]],
    trap_step: int,
    difficulty: int = 2,
    scenario: str | None = None,
) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "start": start,
        "steps": steps,
        "trap_step": trap_step,
        "difficulty": difficulty,
        "scenario": scenario,
    }


_ARITHMETIC_SPECS: list[dict[str, Any]] = [
    _ar("ar_box_apples_1", 50, [("+", 12), ("-", 7), ("*", 2)], trap_step=1, difficulty=2),
    _ar("ar_box_oranges_2", 80, [("-", 25), ("+", 10), ("/", 5)], trap_step=0, difficulty=2),
    _ar("ar_box_screws_3", 100, [("*", 2), ("-", 30), ("+", 15)], trap_step=1, difficulty=2,
        scenario="A hardware shop tracks the contents of a screw bin through the morning."),
    _ar("ar_box_books_4", 36, [("+", 18), ("/", 6), ("*", 3)], trap_step=2, difficulty=3,
        scenario="A library staff member tracks the contents of a returns cart."),
    _ar("ar_box_jars_5", 60, [("-", 15), ("*", 3), ("+", 12)], trap_step=0, difficulty=3,
        scenario="A pickling co-op tracks jars in a holding rack."),
    _ar("ar_box_crayons_6", 80, [("+", 20), ("-", 30), ("*", 2)], trap_step=2, difficulty=2,
        scenario="A primary school art room tracks crayons in a supply box."),
    _ar("ar_box_markers_7", 48, [("/", 4), ("+", 9), ("*", 5)], trap_step=0, difficulty=3,
        scenario="A whiteboard supply closet tracks markers."),
    _ar("ar_box_files_8", 72, [("+", 18), ("-", 40), ("/", 2)], trap_step=1, difficulty=3,
        scenario="An archive room tracks file folders awaiting filing."),
    _ar("ar_box_tools_9", 30, [("*", 4), ("-", 25), ("+", 11)], trap_step=2, difficulty=2,
        scenario="A repair shop tracks loose tools on a workbench."),
    _ar("ar_box_buttons_10", 200, [("-", 75), ("/", 5), ("*", 6)], trap_step=0, difficulty=3,
        scenario="A garment factory tracks buttons in a small parts bin."),
    _ar("ar_box_pencils_11", 84, [("+", 24), ("-", 36), ("/", 4)], trap_step=2, difficulty=3,
        scenario="A test centre tracks pencils in a candidate kit box."),
    _ar("ar_box_corks_12", 144, [("/", 12), ("+", 7), ("*", 3)], trap_step=0, difficulty=4,
        scenario="A winery tracks corks awaiting use in a holding tray."),
    _ar("ar_box_chips_13", 64, [("*", 2), ("-", 50), ("+", 18)], trap_step=2, difficulty=2,
        scenario="A casino backroom tracks chips in a transit box."),
    _ar("ar_box_dice_14", 27, [("*", 3), ("-", 30), ("+", 14)], trap_step=1, difficulty=2,
        scenario="A board-game cafe tracks dice in a sorting tray."),
    _ar("ar_box_ribbons_15", 96, [("-", 24), ("/", 6), ("*", 4)], trap_step=0, difficulty=3,
        scenario="A wrapping station tracks ribbons in a supply tray."),
    _ar("ar_box_eggs_16", 48, [("+", 12), ("-", 30), ("*", 5)], trap_step=2, difficulty=3,
        scenario="A bakery tracks eggs in a holding crate."),
    _ar("ar_box_seeds_17", 100, [("/", 4), ("+", 25), ("*", 2)], trap_step=2, difficulty=3,
        scenario="A nursery tracks seed packets in a shipping bin."),
    _ar("ar_box_tags_18", 72, [("-", 18), ("*", 2), ("+", 16)], trap_step=0, difficulty=2,
        scenario="A shipping desk tracks luggage tags in a queue tray."),
    _ar("ar_box_keys_19", 60, [("*", 3), ("-", 50), ("+", 22)], trap_step=2, difficulty=5,
        scenario="A property manager tracks keys on a daily exchange board."),
    _ar("ar_box_marbles_20", 60, [("+", 30), ("-", 30), ("*", 6)], trap_step=2, difficulty=3,
        scenario="A toy store tracks marbles in a refill jar."),
    _ar("ar_box_petals_21", 84, [("/", 7), ("*", 9), ("-", 24)], trap_step=0, difficulty=4,
        scenario="A florist tracks loose petals in a sample tray."),
    _ar("ar_box_strings_22", 60, [("+", 20), ("-", 30), ("*", 5)], trap_step=2, difficulty=3,
        scenario="A music shop tracks loose guitar strings in a display box."),
    _ar("ar_box_tickets_23", 110, [("-", 30), ("/", 5), ("+", 18)], trap_step=0, difficulty=3,
        scenario="A box office tracks unused tickets in an end-of-show audit."),
    _ar("ar_box_clips_24", 80, [("-", 16), ("/", 4), ("*", 8)], trap_step=2, difficulty=3,
        scenario="An office supply room tracks paper clips in a supply tin."),
    _ar("ar_box_charges_25", 36, [("*", 2), ("-", 40), ("+", 24)], trap_step=2, difficulty=5,
        scenario="A field engineer tracks rechargeable batteries on a service rack."),
]


# --------------------------------------------------------------------- #
# orchestration                                                         #
# --------------------------------------------------------------------- #


@beartype
def build_full_benchmark(seed: int = 0) -> list[BenchmarkItem]:
    """Build the canonical benchmark item list.

    The function is deterministic; ``seed`` is accepted for symmetry
    with the rest of the codebase but currently unused (every spec
    is explicit). It will become meaningful when we add stochastic
    spec generation.
    """
    del seed  # reserved
    items: list[BenchmarkItem] = []

    # ---- CRT
    items.extend(make_many(bat_ball_isomorph, _CRT_RATIO_SPECS))
    items.extend(make_many(widgets_machines_isomorph, _CRT_WORK_RATE_SPECS))
    items.extend(make_many(lily_pad_isomorph, _CRT_EXPGROWTH_SPECS))

    # ---- non-CRT
    items.extend(make_many(base_rate_isomorph, _BASE_RATE_SPECS))
    items.extend(make_many(belief_bias_syllogism, _SYLLOGISM_SPECS))
    items.extend(make_many(anchoring_isomorph, _ANCHORING_SPECS))
    items.extend(make_many(framing_isomorph, _FRAMING_SPECS))
    items.extend(make_many(conjunction_fallacy_isomorph, _CONJUNCTION_SPECS))
    items.extend(make_many(arithmetic_trap_isomorph, _ARITHMETIC_SPECS))

    _assert_invariants(items)
    return items


@beartype
def write_jsonl(items: list[BenchmarkItem], path: str | Path) -> None:
    """Write items to JSONL, one per line, in the order received."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it.to_dict(), ensure_ascii=False))
            fh.write("\n")


def _assert_invariants(items: list[BenchmarkItem]) -> None:
    """Cheap construction-time invariants beyond the formal validator.

    These are NOT a substitute for
    :func:`s1s2.benchmark.validate.validate_benchmark` -- they fire
    early so a developer editing this file gets a stack trace at the
    bad spec, not a wall of validator errors at the end.
    """
    seen_ids: set[str] = set()
    for it in items:
        if it.id in seen_ids:
            raise ValueError(f"duplicate id at construction: {it.id}")
        seen_ids.add(it.id)
    pairs: dict[str, list[BenchmarkItem]] = {}
    for it in items:
        pairs.setdefault(it.matched_pair_id, []).append(it)
    for pid, group in pairs.items():
        if len(group) != 2:
            raise ValueError(
                f"matched_pair_id {pid!r} has {len(group)} primaries; "
                "every primary pair_id must produce exactly one conflict "
                "and one control"
            )
        a, b = group
        if a.difficulty != b.difficulty:
            raise ValueError(
                f"pair {pid}: difficulty mismatch ({a.difficulty} vs {b.difficulty})"
            )
        if a.category != b.category:
            raise ValueError(
                f"pair {pid}: category mismatch ({a.category} vs {b.category})"
            )
