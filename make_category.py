import json
import uuid

# 你的 7 个类别定义
classes = [
    {"name": "static.field.grass", "desc": "Grass area"},
    {"name": "static.driveable_surface.road", "desc": "Concrete path"},
    {"name": "static.terrain.sand", "desc": "Bunker and mud"},
    {"name": "static.fluid.water", "desc": "Water puddle"},
    {"name": "static.vegetation.tree", "desc": "Trees"},
    {"name": "human.pedestrian.adult", "desc": "Person"},
    {"name": "vehicle.glof_cart", "desc": "Standard golf carts used for transporting people or light equipment. Includes both open-top and roofed carts. Usually electric, moving at low speeds."},
    {"name": "vehicle.service.ball_picker", "desc": "Specialized armored golf carts or automated machines used for collecting golf balls. Characterized by protective metal cages/mesh around the driver and a ball-collecting mechanism (roller) at the front."},
    {"name": "vehicle.service.mower", "desc": "Lawn mowers and turf maintenance equipment. Includes ride-on mowers, small tractors, and autonomous robotic mowers (unmanned)."},
    {"name": "vehicle.car", "desc": "Standard automobiles (sedans, SUVs, trucks) that may appear on service roads or parking areas adjacent to the driving range."},
    #{"name": "static.object.obstacle", "desc": "Manmade obstacle"},
    {"name": "static.object.obstacle.pole", "desc": "Vertical thin pole-like objects. Includes light poles, flagsticks, metal net posts, and surveillance poles. Characterized by being tall and thin, prone to visual fragmentation."},
    {"name": "static.object.obstacle.marker", "desc": "Distance marker signs (yardage signs). Includes 50y/100y/150y signs. Typically flat boards or 3D box shapes with numbers displayed."},
    {"name": "static.object.obstacle.crate", "desc": "Ball baskets/crates. Mesh or solid containers placed on the ground for holding balls. Classified as low obstacles, potentially stacked densely."},
    {"name": "static.structure.fence.net", "desc": "High protective netting or mesh fencing surrounding the driving range boundaries. Includes the mesh fabric and its supporting pillars/frames."},
    {"name": "static.structure.building", "desc": "Large static building structures. Includes the clubhouse, surrounding residential houses, maintenance sheds, and other large roofed structures."}
]

category_json = []
for i, cls in enumerate(classes):
    entry = {
        "token": uuid.uuid4().hex, # 自动生成唯一 Token
        "name": cls["name"],
        "description": cls["desc"]
    }
    category_json.append(entry)

# 保存文件
with open('category.json', 'w') as f:
    json.dump(category_json, f, indent=4)
