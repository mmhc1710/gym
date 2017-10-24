from gym.scoreboard.registration import add_task, add_group

# Scoreboard registration
# ==========================
USERNAME='ppaquette'
add_group(
    id= 'super-mario',
    name= 'SuperMario',
    description= '32 levels of the original Super Mario Bros game.'
)

add_task(
    id='{}/meta-SuperMarioBros-v0'.format(USERNAME),
    group='super-mario',
    summary='Compilation of all 32 levels of Super Mario Bros. on Nintendo platform - Screen version.',
)
add_task(
    id='{}/meta-SuperMarioBros-Tiles-v0'.format(USERNAME),
    group='super-mario',
    summary='Compilation of all 32 levels of Super Mario Bros. on Nintendo platform - Tiles version.',
)

for world in range(8):
    for level in range(4):
        add_task(
            id='{}/SuperMarioBros-{}-{}-v0'.format(USERNAME, world + 1, level + 1),
            group='super-mario',
            summary='Level: {}-{} of Super Mario Bros. on Nintendo platform - Screen version.'.format(world + 1, level + 1),
        )
        add_task(
            id='{}/SuperMarioBros-{}-{}-Tiles-v0'.format(USERNAME, world + 1, level + 1),
            group='super-mario',
            summary='Level: {}-{} of Super Mario Bros. on Nintendo platform - Tiles version.'.format(world + 1, level + 1),
        )