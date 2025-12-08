from openalea.plantgl.all import *
from math import *

PANELS, POLES, WIRES, SOIL, SENSORS = 1, 2, 3, 4, 5
# Reflectance_Up, Transmittance_Up, Reflectance_Down, Transmittance_Down


def generate_plots():
    panel = QuadSet(
        [(0, 0, 0), (1.0, 0, 0), (1.0, 1.93, 0), (0, 1.93, 0)],
        [list(range(4)), panelcolor],
    )

    panel3 = Translated(
        0,
        0,
        5.12,
        AxisRotated(
            Vector3.OX,
            radians(15),
            Group(
                [panel, Translated(1.075, 0, 0, panel), Translated(2.15, 0, 0, panel)]
            ),
        ),
    )
    panel3 = [
        Translated(0, 0, 5.12, AxisRotated(Vector3.OX, radians(15), p))
        for p in [panel, Translated(1.075, 0, 0, panel), Translated(2.15, 0, 0, panel)]
    ]

    row = [Translated(7.5 * i, 0, 0, p) for i in range(6) for p in panel3]

    panelmatrix = [
        Translated(3.7 if j % 2 == 0 else 7.45, 3 * j, 0, geom)
        for j in range(5)
        for geom in row
    ]

    scene = Scene([Shape(panel, id=PANELS) for panel in panelmatrix])

    borderpole = Cylinder(0.125, 6.81, solid=False)

    # 7.58 à gauche
    # 7.02 à droite
    leftangle = 7.58
    rightangle = 7.02

    leftborderpoles = [
        Translated(3.4, 0.96 + 3 * i, 0, p)
        for p in [
            AxisRotated(
                Vector3.OY,
                radians(-30),
                AxisRotated(Vector3.OX, radians(leftangle), borderpole),
            ),
            AxisRotated(
                Vector3.OY,
                radians(-30),
                AxisRotated(Vector3.OX, radians(-rightangle), borderpole),
            ),
        ]
        for i in range(5)
    ]

    rightborderpoles = [
        Translated(48.4, 0.96 + 3 * i, 0, p)
        for p in [
            AxisRotated(
                Vector3.OY,
                radians(30),
                AxisRotated(Vector3.OX, radians(leftangle), borderpole),
            ),
            AxisRotated(
                Vector3.OY,
                radians(30),
                AxisRotated(Vector3.OX, radians(-rightangle), borderpole),
            ),
        ]
        for i in range(5)
    ]

    borderpoles = leftborderpoles + rightborderpoles

    centralpole = Cylinder(0.125 / 2, 5.9, solid=False)

    leftcentralpoles = [
        Translated(18.4, 0.96 + 3 * i, 0, p)
        for p in [
            AxisRotated(Vector3.OX, radians(leftangle), centralpole),
            AxisRotated(Vector3.OX, radians(-rightangle), centralpole),
        ]
        for i in range(5)
    ]

    rightcentralpoles = [
        Translated(33.4, 0.96 + 3 * i, 0, p)
        for p in [
            AxisRotated(Vector3.OX, radians(leftangle), centralpole),
            AxisRotated(Vector3.OX, radians(-rightangle), centralpole),
        ]
        for i in range(5)
    ]

    scene += Scene(
        [
            Shape(pole, id=POLES)
            for pole in borderpoles + leftcentralpoles + rightcentralpoles
        ]
    )
    return scene


##### TILES FOR RAY TRACING
########### COLORS FOR FLOORS
ricecol = Material("#88AA59", Color3(80,100,45))
groundcol = Material("#B68354", Color3(135,100,80))
sensorcol = Material("#94B5DA", Color3(148,181,218))

NBCOL = 100
NBLIG = 30
MAPLENGTH = 50.0  # m
MAPWIDTH = 15.0   # m

def sensorsids():
    return [(col, rank)  for rank in range(NBLIG) for col in range(NBCOL)]

IDDECALPOWER = 4
IDDECAL = pow(10, IDDECALPOWER)

def position2id(col, rank):
    return int(rank*pow(10, IDDECALPOWER*2) + col*IDDECAL)

def id2position(id):
    col = (id // IDDECAL) % IDDECAL
    rank = id // pow(10,IDDECALPOWER*2)
    return int(col), int(rank)

def sensorpositions(height = 0):
    WCOL = MAPLENGTH / NBCOL
    WROW = MAPWIDTH / NBLIG
    return [(position2id(col, rank),(WCOL*col,WROW*rank,height)) for col, rank in sensorsids()]

def sensorgeometry(height = 0):
    WCOL = MAPLENGTH / NBCOL
    WROW = MAPWIDTH / NBLIG
    tile = QuadSet([(-WCOL/2,0,0),(WCOL/2,0,0),(WCOL/2,WROW,0),(-WCOL/2,WROW,0)], [list(range(4))])
    floor = [Shape(Translated(position,tile), groundcol, id) for id, position in sensorpositions(height)]
    ########### GROUND
    return  Scene(floor)



if __name__ == "__main__":
    Viewer.display(generate_plots())
