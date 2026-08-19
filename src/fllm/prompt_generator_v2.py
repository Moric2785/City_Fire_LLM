import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging


@dataclass
class FireIncidentData:
    """Fire incident data structure"""
    incident_key: str
    building_info: Dict
    weather_info: Dict
    fire_info: Dict
    community_context: Dict
    fire_spread_label: int  # 2, 3, 4, 5


class FirePromptGenerator:
    """Fire incident prompt generator"""

    def __init__(self, codelookup_path: str = "./data/codelookup14.txt"):
        """Initialize prompt generator"""
        # Fire spread class mapping (3 classes: 2,3,4, where class 4 includes original 4 and 5)
        self.fire_spread_classes = {
            2: "fire within the room",
            3: "within the floor",
            4: "beyond the floor"
        }

        # Initialize encoding mappings (fallback maps for fields with no usable *_desc column)
        self._init_code_mappings()

        # Load NFIRS official codelookup table (fieldid -> {code: desc})
        self.CODE_LOOKUP = self._load_code_lookup(codelookup_path)

    def _load_code_lookup(self, path: str) -> Dict[str, Dict[str, str]]:
        """Load NFIRS official codelookup14.txt file and group into a {fieldid: {code: desc}} dict.
        On failure, logs a warning and returns an empty dict so downstream lookups safely fall back to 'Unknown'."""
        try:
            lut = pd.read_csv(path, sep='^', engine='python', quotechar='"')
        except Exception as e:
            logging.warning(f"Failed to load codelookup file {path}: {e}")
            return {}

        lookup: Dict[str, Dict[str, str]] = {}
        for fieldid, sub in lut.groupby('fieldid'):
            lookup[fieldid] = dict(zip(sub['code_value'].astype(str), sub['code_descr']))
        return lookup

    def _lookup_from_codefile(self, fieldid: str, code_val: Any) -> str:
        """Look up a code's description from codelookup14.txt for a given NFIRS fieldid.
        Handles zero-padding mismatches (e.g. data has '0', lookup table has '000').
        Returns 'Unknown' if the table is missing, the code is missing, or no match is found."""
        table = self.CODE_LOOKUP.get(fieldid)
        if not table or self._is_missing(code_val):
            return 'Unknown'

        v = str(code_val).strip()
        for candidate in (v, v.zfill(3), v.lstrip('0') or '0'):
            if candidate in table and not self._is_missing(table[candidate]):
                return str(table[candidate]).strip()
        return 'Unknown'

    def _get_desc(self, row: pd.Series, base_col: str, code_map: Dict = None, desc_col: str = None) -> str:
        """Get a display description for a field:
        1) Prefer reading directly from a '{base_col}_desc' column (or explicit desc_col override
           when the code column and desc column don't share the same base name).
        2) If missing/absent, fall back to code_map lookup on the raw '{base_col}' column
           (no more '_new' fallback — read the raw NFIRS column directly).
        3) Otherwise return 'Unknown'."""
        desc_col = desc_col or f"{base_col}_desc"
        if desc_col in row.index:
            desc_val = row.get(desc_col)
            if not self._is_missing(desc_val):
                return self._clean_str(desc_val)

        code_val = row.get(base_col, 'Unknown')
        if code_map is not None:
            return self.map_code(code_map, code_val, 'Unknown')
        return 'Unknown'

    def _init_code_mappings(self):
        """Initialize various encoding mappings (NFIRS official code tables)"""

        self.FIRE_ORIG_MAP = {
            '0': 'Other',
            '1': 'Corridor, mall',
            '10': 'Assembly or sales area, other',
            '11': 'Arena, assembly area w/ fixed seats - 100+ persons',
            '12': 'Assembly area without fixed seats - 100+ persons',
            '13': 'Assembly area - less than 100 persons',
            '14': 'Common room, den, family room, living room, lounge',
            '15': 'Sales area, showroom (exclude display window)',
            '17': 'Swimming pool',
            '2': 'Exterior stairway, ramp, or fire escape',
            '20': 'Function area, other',
            '21': 'Bedroom - < 5 persons; included are jail or prison',
            '22': 'Bedroom - 5+ persons; including barrack/dormitory',
            '23': 'Dining room, cafeteria bar area, beverage service',
            '24': 'Cooking area, kitchen',
            '25': 'Bathroom, checkroom, lavatory, locker room',
            '26': 'Laundry area, wash house (laundry)',
            '27': 'Office',
            '28': 'Personal service area, barber/beauty salon area',
            '3': 'Interior stairway or ramp',
            '30': 'Technical processing areas, other',
            '31': 'Laboratory',
            '32': 'Dark room, photography area, or printing area',
            '35': 'Computer room, control room or center',
            '36': 'Stage area - performance, basketball court, boxing',
            '37': 'Projection room, spotlight area',
            '38': 'Processing/manufacturing area, workroom',
            '4': 'Escalator - exterior, interior',
            '40': 'Storage area, other',
            '41': 'Storage room, area, tank, or bin',
            '42': 'Closet',
            '43': 'Storage: supplies or tools; dead storage',
            '44': 'Records storage room, storage vault',
            '45': 'Shipping/receiving area; loading area, dock or bay',
            '46': 'Chute/container - trash, rubbish, waste',
            '47': 'Vehicle storage area; garage, carport',
            '5': 'Entrance way, lobby',
            '50': 'Service facilities, other',
            '52': 'Conduit, pipe, utility, or ventilation shaft',
            '53': 'Light shaft',
            '54': 'Chute; laundry or mail, excluding trash chutes',
            '55': 'Duct: hvac, cable, exhaust, heating, or AC',
            '56': 'Display window',
            '58': 'Conveyor',
            '60': 'Equipment or service area, other',
            '61': 'Machinery room or area; elevator machinery room',
            '62': 'Heating room or area, water heater area',
            '63': 'Switchgear area, transformer vault',
            '65': 'Maintenance shop or area, paint shop or area',
            '70': 'Structural area, other',
            '71': 'Substructure area or space, crawl space',
            '72': 'Exterior balcony, unenclosed porch',
            '73': 'Ceiling & floor assembly, crawl space b/t stories',
            '74': 'Attic: vacant, crawl space above top story, cupola',
            '75': 'Wall assembly',
            '76': 'Wall surface: exterior',
            '77': 'Roof surface: exterior',
            '78': 'Awning',
            '9': 'Egress/exit, other',
        }

        self.PROP_USE_MAP = {
            '0': 'Property Use, other',
            '100': 'Assembly, other',
            '110': 'Fixed use recreation places, other',
            '111': 'Bowling alley',
            '116': 'Swimming facility: indoor or outdoor',
            '120': 'Variable use amusement, recreation places',
            '121': 'Ballroom, gymnasium',
            '123': 'Stadium, arena',
            '124': 'Playground',
            '130': 'Places of worship, funeral parlors',
            '131': 'Church, mosque, synagogue, temple, chapel',
            '134': 'Funeral parlor',
            '140': 'Clubs, other',
            '141': 'Athletic/health club',
            '142': 'Clubhouse',
            '143': 'Yacht Club',
            '144': 'Casino, gambling clubs',
            '150': 'Public or government, other',
            '160': 'Eating, drinking places',
            '161': 'Restaurant or cafeteria',
            '162': 'Bar or nightclub',
            '180': 'Studio/theater, other',
            '186': 'Film/movie production studio',
            '200': 'Educational, other',
            '210': 'Schools, non-adult',
            '211': 'Preschool',
            '213': 'Elementary school, including kindergarten',
            '215': 'High school/junior high school/middle school',
            '241': 'Adult education center, college classroom',
            '254': 'Day care, in commercial property',
            '255': 'Day care, in residence, licensed',
            '311': '24-hour care Nursing homes, 4 or more persons',
            '321': 'Mental retardation/development disability facility',
            '322': 'Alcohol or substance abuse recovery center',
            '331': 'Hospital - medical or psychiatric',
            '340': 'Clinics, Doctors offices, hemodialysis centers',
            '341': 'Clinic, clinic-type infirmary',
            '342': "Doctor, dentist or oral surgeon's office",
            '361': 'Jail, prison (not juvenile)',
            '363': 'Reformatory, juvenile detention center',
            '365': 'Police station',
            '400': 'Residential, other',
            '419': '1 or 2 family dwelling',
            '429': 'Multifamily dwellings',
            '439': 'Boarding/rooming house, residential hotels',
            '449': 'Hotel/motel, commercial',
            '459': 'Residential board and care',
            '460': 'Dormitory type residence, other',
            '462': 'Sorority house, fraternity house',
            '464': 'Barracks, dormitory',
            '500': 'Mercantile, business, other',
            '511': 'Convenience store',
            '519': 'Food and beverage sales, grocery store',
            '529': 'Textile, wearing apparel sales',
            '539': 'Household goods, sales, repairs',
            '549': 'Specialty shop',
            '557': 'Personal service, including barber & beauty shops',
            '559': 'Recreational, hobby, home repair sales, pet store',
            '564': 'Laundry, dry cleaning',
            '569': 'Professional supplies, services',
            '571': 'Service station, gas station',
            '579': 'Motor vehicle or boat sales, services, repair',
            '580': 'General retail, other',
            '581': 'Department or discount store',
            '592': 'Bank',
            '593': 'Office:  veterinary or research',
            '596': 'Post office or mailing firms',
            '599': 'Business office',
            '600': 'Utility, defense, agriculture, mining, other',
            '615': 'Electric generating plant',
            '629': 'Laboratory or science lababoratory',
            '635': 'Computer center',
            '639': 'Communications center',
            '645': 'Flammable liquid distribution, pipeline, flammable',
            '647': 'Water utility',
            '648': 'Sanitation utility',
            '655': 'Crops or orchard',
            '659': 'Livestock production',
            '669': 'Forest, timberland, woodland',
            '700': 'Manufacturing, processing',
            '800': 'Storage, other',
            '807': 'Outside material storage area',
            '808': 'Outbuilding or shed',
            '816': 'Grain elevator, silo',
            '819': 'Livestock, poultry storage',
            '839': 'Refrigerated storage',
            '849': 'Outside storage tank',
            '880': 'Vehicle storage, other',
            '881': 'Parking garage, (detached residential garage)',
            '882': 'Parking garage, general vehicle',
            '888': 'Fire station',
            '891': 'Warehouse',
            '898': 'Dock, marina, pier, wharf',
            '899': 'Residential or self storage units',
            '900': 'Outside or special property, other',
            '919': 'Dump, sanitary landfill',
            '926': 'Outbuilding, protective shelter',
            '960': 'Street, other',
            '961': 'Highway or divided highway',
            '962': 'Residential street, road or residential driveway',
            '963': 'Street or road in commercial area',
            '965': 'Vehicle parking area',
            '984': 'Industrial plant yard - area',
        }

        self.AREA_ORIG_MAP = {
            '0': 'Other',
            '1': 'Corridor, mall',
            '10': 'Assembly or sales area, other',
            '11': 'Arena, assembly area w/ fixed seats - 100+ persons',
            '12': 'Assembly area without fixed seats - 100+ persons',
            '13': 'Assembly area - less than 100 persons',
            '14': 'Common room, den, family room, living room, lounge',
            '15': 'Sales area, showroom (exclude display window)',
            '17': 'Swimming pool',
            '2': 'Exterior stairway, ramp, or fire escape',
            '20': 'Function area, other',
            '21': 'Bedroom - < 5 persons; included are jail or prison',
            '22': 'Bedroom - 5+ persons; including barrack/dormitory',
            '23': 'Dining room, cafeteria bar area, beverage service',
            '24': 'Cooking area, kitchen',
            '25': 'Bathroom, checkroom, lavatory, locker room',
            '26': 'Laundry area, wash house (laundry)',
            '27': 'Office',
            '28': 'Personal service area, barber/beauty salon area',
            '3': 'Interior stairway or ramp',
            '30': 'Technical processing areas, other',
            '31': 'Laboratory',
            '32': 'Dark room, photography area, or printing area',
            '35': 'Computer room, control room or center',
            '36': 'Stage area - performance, basketball court, boxing',
            '37': 'Projection room, spotlight area',
            '38': 'Processing/manufacturing area, workroom',
            '4': 'Escalator - exterior, interior',
            '40': 'Storage area, other',
            '41': 'Storage room, area, tank, or bin',
            '42': 'Closet',
            '43': 'Storage: supplies or tools; dead storage',
            '44': 'Records storage room, storage vault',
            '45': 'Shipping/receiving area; loading area, dock or bay',
            '46': 'Chute/container - trash, rubbish, waste',
            '47': 'Vehicle storage area; garage, carport',
            '5': 'Entrance way, lobby',
            '50': 'Service facilities, other',
            '52': 'Conduit, pipe, utility, or ventilation shaft',
            '53': 'Light shaft',
            '54': 'Chute; laundry or mail, excluding trash chutes',
            '55': 'Duct: hvac, cable, exhaust, heating, or AC',
            '56': 'Display window',
            '58': 'Conveyor',
            '60': 'Equipment or service area, other',
            '61': 'Machinery room or area; elevator machinery room',
            '62': 'Heating room or area, water heater area',
            '63': 'Switchgear area, transformer vault',
            '65': 'Maintenance shop or area, paint shop or area',
            '70': 'Structural area, other',
            '71': 'Substructure area or space, crawl space',
            '72': 'Exterior balcony, unenclosed porch',
            '73': 'Ceiling & floor assembly, crawl space b/t stories',
            '74': 'Attic: vacant, crawl space above top story, cupola',
            '75': 'Wall assembly',
            '76': 'Wall surface: exterior',
            '77': 'Roof surface: exterior',
            '78': 'Awning',
            '9': 'Egress/exit, other',
        }

        self.HEAT_SOURC_MAP = {
            '0': 'Heat source: other',
            '10': 'Heat from powered equipment, other',
            '11': 'Spark, ember or flame from operating equipment',
            '12': 'Radiated, conducted heat from operating equipment',
            '13': 'Arcing',
            '40': 'Hot or smoldering object, other',
            '41': 'Heat, spark from friction',
            '42': 'Molten, hot material',
            '43': 'Hot ember or ash',
            '50': 'Explosive, fireworks, other',
            '51': 'Munitions',
            '54': 'Fireworks',
            '56': 'Incendiary device',
            '60': 'Heat from other open flame or smoking materials',
            '61': 'Cigarette',
            '62': 'Pipe or cigar',
            '63': 'Heat from undetermined smoking material',
            '64': 'Match',
            '65': 'Cigarette lighter',
            '66': 'Candle',
            '68': 'Backfire from internal combustion engine',
            '69': 'Flame/torch used for lighting',
            '70': 'Chemical, natural heat source, other',
            '71': 'Sunlight',
            '72': 'Chemical reaction',
            '73': 'Lightning',
            '74': 'Other static discharge',
            '97': 'Multiple heat sources including multiple ignitions',
        }

        self.FIRST_IGN_MAP = {
            '0': 'Item First Ignited, Other',
            '10': 'Structural component or finish, other',
            '11': 'Exterior roof covering or finish',
            '12': 'Exterior wall covering or finish',
            '13': 'Exterior trim, including doors',
            '14': 'Floor covering or rug/carpet/mat',
            '15': 'Interior wall covering excluding drapes, etc.',
            '16': 'Interior ceiling cover or finish',
            '17': 'Structural member or framing',
            '18': 'Insulation within structural area',
            '20': 'Furniture, utensils, other',
            '21': 'Upholstered sofa, chair, vehicle seats',
            '22': 'Non-upholstered chair, bench',
            '23': 'Cabinetry (including built-in)',
            '24': 'Ironing board',
            '25': 'Appliance housing or casing',
            '26': 'Household utensils',
            '30': 'Soft goods, wearing apparel, other',
            '31': 'Mattress, pillow',
            '32': 'Bedding; blanket, sheet, comforter',
            '33': 'Linen; other than bedding',
            '34': 'Wearing apparel not on a person',
            '35': 'Wearing apparel on a person',
            '36': 'Curtains, blinds, drapery, tapestry',
            '37': 'Goods not made up, including fabrics & yard goods',
            '38': 'Luggage',
            '40': 'Adornment, recreational material, signs, other',
            '41': 'Christmas tree',
            '42': 'Decoration',
            '43': 'Sign, including outdoor signs such as billboards',
            '44': 'Chips, including wood chips',
            '45': 'Toy or game',
            '46': 'Awning, canopy',
            '47': 'Tarpaulin or tent',
            '50': 'Storage supplies, other',
            '51': 'Box, carton, bag, basket, barrel',
            '52': 'Material being used to make a product',
            '53': 'Pallet, skid (empty)',
            '54': 'Cord, rope, twine',
            '55': 'Packing, wrapping material',
            '56': 'Baled goods or material',
            '57': 'Bulk storage',
            '58': 'Palletized material, material stored on pallets.',
            '59': 'Rolled, wound material (paper, fabric)',
            '60': 'Liquids, piping, filters, other',
            '61': 'Atomized liquid, vaporized liquid, aerosol.',
            '62': 'Flammable liquid/gas - in/from engine or burner',
            '63': 'Flammable liquid/gas - in/from final container',
            '64': 'Flammable liquid/gas in container or pipe',
            '65': 'Flammable liquid/gas - uncontained',
            '66': 'Pipe, duct, conduit or hose',
            '67': 'Pipe, duct, conduit, hose covering',
            '68': 'Filter, including evaporative cooler pads',
            '70': 'Organic materials, other',
            '71': 'Agricultural crop, including fruits and vegetables',
            '72': 'Light vegetation - not crop, including grass',
            '73': 'Heavy vegetation - not crop, including trees',
            '74': 'Animal living or dead',
            '76': 'Cooking materials, including edible materials',
            '77': 'Feathers or fur, not on bird or animal',
            '81': 'Electrical wire, cable insulation',
            '82': 'Transformer, including transformer fluids',
            '83': 'Conveyor belt, drive belt, V-belt',
            '84': 'Tire',
            '86': 'Fence, pole',
            '87': 'Fertilizer',
            '88': 'Pyrotechnics, explosives',
            '91': 'Book',
            '92': 'Magazine, newspaper, writing paper',
            '93': 'Adhesive',
            '94': 'Dust, fiber, lint, including sawdust and excelsior',
            '95': 'Film, residue, including paint & resin',
            '96': 'Rubbish, trash, or waste',
            '97': 'Oily rags',
            '99': 'Multiple items first ignited',
        }

        self.TYPE_MAT_MAP = {
            '0': 'Type of material first ignited, other',
            '10': 'Flammable gas, other',
            '11': 'Natural gas',
            '12': 'LP gas',
            '14': 'Acetylene',
            '15': 'Hydrogen',
            '20': 'Flammable or combustible liquid, other',
            '21': 'Ether, pentane type flammable liquid',
            '22': 'JP-4 jet fuel & methyl ethyl ketone type flammable',
            '23': 'Gasoline',
            '24': 'Turpentine, butyl alcohol type flammable liquid',
            '25': 'Kerosene, No.1 and 2 fuel oil, diesel type',
            '26': 'Cottonseed oil, creosote oil type combustible',
            '27': 'Cooking oil, transformer or lubricating oil',
            '28': 'Ethanol',
            '30': 'Volatile solid or chemical, other',
            '31': 'Fat, grease, butter, margarine, lard',
            '33': 'Polish, paraffin, wax',
            '34': 'Adhesive, resin, tar, glue, asphalt, pitch',
            '35': 'Paint, varnish - applied',
            '36': 'Combustible metal, included are magnesium',
            '37': 'Solid chemical, included are explosives',
            '38': 'Radioactive material',
            '41': 'Plastic',
            '50': 'Natural product, other',
            '51': 'Rubber, excluding synthetic rubbers',
            '52': 'Cork',
            '53': 'Leather',
            '54': 'Hay, straw',
            '55': 'Grain, natural fiber,  (preprocess)',
            '56': 'Coal, coke, briquettes, peat',
            '57': 'Food, starch, excluding fat and grease (Code 31)',
            '58': 'Tobacco',
            '60': 'Wood or paper, processed, other',
            '61': 'Wood chips, sawdust, shavings',
            '62': 'Round timber, including round posts, poles',
            '63': 'Sawn wood, including all finished lumber',
            '64': 'Plywood',
            '65': 'Fiberboard, particleboard, and hardboard',
            '66': 'Wood pulp',
            '67': 'Paper, including cellulose, waxed paper',
            '68': 'Cardboard',
            '70': 'Fabric, textile, fur, other',
            '71': 'Fabric, fiber, cotton, blends, rayon, wool',
            '74': 'Fur, silk, other fabric.',
            '77': 'Plastic coated fabric',
            '80': 'Material compounded with oil, other',
            '81': 'Linoleum',
            '82': 'Oilcloth',
            '86': 'Asphalt treated material',
            '99': 'Multiple types of material',
        }

        self.ACT_TAK1_MAP = {
            '0': 'Action taken, other',
            '10': 'Fire, other',
            '11': 'Extinguish',
            '12': 'Salvage & overhaul',
            '14': 'Contain fire (wildland)',
            '16': 'Control fire (wildland)',
            '20': 'Search & rescue, other',
            '21': 'Search',
            '22': 'Rescue, remove from harm',
            '23': 'Extricate, disentangle',
            '24': 'Recover body',
            '30': 'Emergency medical services, other',
            '31': 'Provide first aid & check for injuries',
            '32': 'Provide basic life support (BLS)',
            '33': 'Provide advanced life support (ALS)',
            '34': 'Transport person',
            '40': 'Hazardous condition, other',
            '41': 'Identify, analyze hazardous materials',
            '43': 'Hazardous materials spill control and confinement',
            '45': 'Remove hazard',
            '50': 'Fires, rescues & hazardous conditions, other',
            '51': 'Ventilate',
            '52': 'Forcible entry',
            '53': 'Evacuate area',
            '54': 'Determine if materials are non-hazardous',
            '55': 'Establish safe area',
            '56': 'Provide air supply',
            '58': 'Operate apparatus or vehicle',
            '62': 'Restore sprinkler or fire protection system',
            '63': 'Restore fire alarm system',
            '64': 'Shut down system',
            '66': 'Remove water',
            '70': 'Assistance, other',
            '73': 'Provide manpower',
            '74': 'Provide apparatus',
            '75': 'Provide equipment',
            '76': 'Provide water',
            '80': 'Information, investigation & enforcement, other',
            '81': 'Incident command',
            '82': 'Notify other agencies.',
            '83': 'Provide information to public or media',
            '84': 'Refer to proper authority',
            '85': 'Enforce code',
            '86': 'Investigate',
            '87': 'Investigate fire out on arrival',
            '90': 'Fill-in, standby, other',
            '92': 'Standby',
        }

        self.FACT_IGN1_MAP = {
            '0': 'Other factor contributed to ignition',
            '10': 'Misuse of material or product, other',
            '11': 'Abandoned or discarded materials or products',
            '12': 'Heat source too close to combustibles.',
            '13': 'Cutting, welding too close to combustible',
            '14': 'Flammable liquid or gas spilled',
            '15': 'Improper fueling technique',
            '16': 'Flammable liquid used to kindle fire',
            '17': 'Washing part, painting with flammable liquid',
            '18': 'Improper container or storage',
            '19': 'Playing with heat source',
            '20': 'Mechanical failure, malfunction, other',
            '21': 'Automatic control failure',
            '22': 'Manual control failure',
            '23': 'Leak or break',
            '25': 'Worn out',
            '26': 'Backfire',
            '27': 'Improper fuel used',
            '30': 'Electrical failure, malfunction, other',
            '31': 'Water caused short-circuit arc',
            '32': 'Short circuit arc from mechanical damage',
            '33': 'Short circuit arc from defective, worn insulation',
            '34': 'Unspecified short-circuit arc',
            '35': 'Arc from faulty contact, broken conductor',
            '36': 'Arc, spark from operating equipment',
            '37': 'Fluorescent light ballast',
            '40': 'Design/Manufacture/Installation Deficiency, other',
            '41': 'Design deficiency',
            '42': 'Construction deficiency',
            '43': 'Installation deficiency',
            '44': 'Manufacturing deficiency',
            '50': 'Operational deficiency, other',
            '51': 'Collision, knock down, run over, turn over',
            '52': 'Accidentally turned on, not turned off',
            '53': 'Equipment unattended',
            '54': 'Equipment overloaded',
            '55': 'Failure to clean',
            '56': 'Improper startup',
            '57': 'Equipment used for not intended purpose',
            '58': 'Equipment not being operated properly',
            '60': 'Natural condition, other',
            '61': 'High wind',
            '62': 'Storm',
            '63': 'High water including floods',
            '66': 'Animal',
        }

        self.SUP_FAC1_MAP = {
            '0': 'Fire supression factor, other',
            '100': 'Building construction or design factors, other',
            '112': 'Roof collapse',
            '113': 'Roof assembly combustible',
            '121': 'Ceiling collapse',
            '125': 'Holes or openings in walls or ceilings',
            '131': 'Wall collapse',
            '132': 'Difficult to ventilate',
            '134': 'Combustible interior finish',
            '137': 'Balloon construction',
            '138': 'Internal arrangement of partitions',
            '139': 'Internal arrangement of stock or contents',
            '141': 'Floor collapse',
            '151': 'Lack of fire barrier walls or doors',
            '161': 'Attic undivided',
            '166': 'Insulation combustible',
            '173': 'Stairwell not enclosed',
            '176': 'Ducts: vertical',
            '177': 'Chute: rubbish, garbage, laundry',
            '183': 'Composite roof/floor sheathing construction',
            '185': 'Wood truss construction',
            '187': 'Fixed burglar protection assemblies (bars, grills',
            '200': 'Act or omission, other',
            '213': 'Doors left open or outside door unsecured',
            '222': 'Illegal and clandestine drug operation',
            '232': 'Intoxication, drugs or alcohol',
            '254': 'Persons interfered with operations',
            '283': 'Accelerant used',
            '300': 'Building contents, other',
            '311': 'Aisles blocked or improper width',
            '312': 'Significant/unusual fuel load structure components',
            '313': 'Significant/unusual fuel load from contents',
            '315': 'Significant fuel load from man-made condition.',
            '316': 'Storage, improper',
            '324': 'Hazardous chemical, corrosive material, or oxidize',
            '325': 'Flammable/combustible liquid hazard',
            '327': 'Explosives hazard present',
            '341': 'Natural or other lighter than air gas present',
            '342': 'Liquefied Petroleum (LPG) gas present',
            '400': 'Delays, other',
            '411': 'Delayed detection of fire',
            '412': 'Delayed reporting of fire',
            '413': 'Alarm system malfunction',
            '415': 'Alarm System inappropriately shut off',
            '424': 'Information incomplete or incorrect',
            '425': 'Communications problem',
            '431': 'Blocked  or obstructed roadway',
            '434': 'Poor or no access for fire department apparatus',
            '435': 'Traffic delay',
            '436': 'Trouble finding location',
            '437': 'Size, height, or other building characteristic',
            '438': 'Power lines down/arcing',
            '443': 'Poor access for firefighters',
            '444': 'Secured area',
            '448': 'Locked or jammed doors',
            '451': 'Apparatus failure before arrival at incident',
            '452': 'Hydrants inoperative',
            '481': 'Closest apparatus unavailable',
            '510': 'Automatic fire supression system problem.',
            '520': 'Automatic sprinkler, standpipe connection problem',
            '531': 'Water supply inadequate: private',
            '532': 'Water supply inadequate: public',
            '561': 'Failure of rated fire protection assembly',
            '600': 'Egress/exit problem, other',
            '613': 'Window type impedes egress',
            '621': 'Young occupants',
            '622': 'Elderly occupants',
            '623': 'Physically disabled occupants',
            '700': 'Natural conditions, other',
            '711': 'Drought or low fuel moisture',
            '712': 'Humidity low',
            '713': 'Humidity high',
            '714': 'Temperature: low',
            '715': 'Temperature: high',
            '721': 'Fog',
            '722': 'Flooding',
            '723': 'Ice',
            '724': 'Rain',
            '725': 'Snow',
            '732': 'Wind, including hurricanes or tornadoes',
            'UUU': 'Undetermined (conversion only)',
        }

        self.AES_PRES_MAP = {
            '1': 'Present',
            '2': 'Partial system present',
            'N': 'None Present',
        }

        self.DETECTOR_MAP = {
            '1': 'Detectors Present',
            'N': 'None Present',
        }

        self.CAUSE_IGN_MAP = {
            '1': 'Intentional',
            '2': 'Unintentional',
            '3': 'Failure of equipment or heat source',
            '4': 'Act of nature',
            '5': 'Cause under investigation',
            'U': 'Cause undetermined after investigation',
        }

        self.HUM1_MAP = {
            '1': 'Asleep',
            '2': 'Possibly impaired by alcohol or drugs',
            '3': 'Unattended or unsupervised person',
            '4': 'Possibly mentally disabled',
            '5': 'Physically disabled',
            '6': 'Multiple persons involved',
            '7': 'Age was a factor',
        }

        self.STRUC_TYPE_MAP = {
            '0': 'Structure type, other',
            '1': 'Enclosed building',
            '2': 'Fixed portable or mobile structure',
            '3': 'Open structure',
            '4': 'Air supported structure',
            '5': 'Tent',
            '6': 'Open platform',
            '7': 'Underground structure work areas',
            '8': 'Connective structure',
        }

        self.SEASON_MAP = {
            1: 'winter', 2: 'winter',  # 1月、2月
            3: 'spring', 4: 'spring', 5: 'spring',  # 3-5月
            6: 'summer', 7: 'summer', 8: 'summer',  # 6-8月
            9: 'fall', 10: 'fall', 11: 'fall',  # 9-11月
            12: 'winter'  # 12月
        }

        self.TIME_PERIOD_MAP = {
            'morning': (6, 11),    # 6:00-11:59
            'afternoon': (12, 17), # 12:00-17:59
            'evening': (18, 21),   # 18:00-21:59
            'night': (22, 5)       # 22:00-5:59
        }

        self.ITEM_SPRD_MAP = {
            '0': 'Item First Ignited, Other',
            '10': 'Structural component or finish, other',
            '11': 'Exterior roof covering or finish',
            '12': 'Exterior wall covering or finish',
            '13': 'Exterior trim, including doors',
            '14': 'Floor covering or rug/carpet/mat',
            '15': 'Interior wall covering excluding drapes, etc.',
            '16': 'Interior ceiling cover or finish',
            '17': 'Structural member or framing',
            '18': 'Insulation within structural area',
            '20': 'Furniture, utensils, other',
            '21': 'Upholstered sofa, chair, vehicle seats',
            '22': 'Non-upholstered chair, bench',
            '23': 'Cabinetry (including built-in)',
            '25': 'Appliance housing or casing',
            '26': 'Household utensils',
            '30': 'Soft goods, wearing apparel, other',
            '31': 'Mattress, pillow',
            '32': 'Bedding; blanket, sheet, comforter',
            '33': 'Linen; other than bedding',
            '34': 'Wearing apparel not on a person',
            '35': 'Wearing apparel on a person',
            '36': 'Curtains, blinds, drapery, tapestry',
            '37': 'Goods not made up, including fabrics & yard goods',
            '40': 'Adornment, recreational material, signs, other',
            '41': 'Christmas tree',
            '42': 'Decoration',
            '43': 'Sign, including outdoor signs such as billboards',
            '45': 'Toy or game',
            '46': 'Awning, canopy',
            '50': 'Storage supplies, other',
            '51': 'Box, carton, bag, basket, barrel',
            '52': 'Material being used to make a product',
            '53': 'Pallet, skid (empty)',
            '55': 'Packing, wrapping material',
            '57': 'Bulk storage',
            '58': 'Palletized material, material stored on pallets.',
            '59': 'Rolled, wound material (paper, fabric)',
            '60': 'Liquids, piping, filters, other',
            '61': 'Atomized liquid, vaporized liquid, aerosol.',
            '62': 'Flammable liquid/gas - in/from engine or burner',
            '63': 'Flammable liquid/gas - in/from final container',
            '64': 'Flammable liquid/gas in container or pipe',
            '65': 'Flammable liquid/gas - uncontained',
            '66': 'Pipe, duct, conduit or hose',
            '67': 'Pipe, duct, conduit, hose covering',
            '70': 'Organic materials, other',
            '71': 'Agricultural crop, including fruits and vegetables',
            '72': 'Light vegetation - not crop, including grass',
            '73': 'Heavy vegetation - not crop, including trees',
            '75': 'Human living or dead',
            '76': 'Cooking materials, including edible materials',
            '81': 'Electrical wire, cable insulation',
            '82': 'Transformer, including transformer fluids',
            '83': 'Conveyor belt, drive belt, V-belt',
            '84': 'Tire',
            '86': 'Fence, pole',
            '88': 'Pyrotechnics, explosives',
            '91': 'Book',
            '92': 'Magazine, newspaper, writing paper',
            '93': 'Adhesive',
            '94': 'Dust, fiber, lint, including sawdust and excelsior',
            '95': 'Film, residue, including paint & resin',
            '96': 'Rubbish, trash, or waste',
            '97': 'Oily rags',
            '99': 'Multiple items first ignited',
        }

        # MAT_SPRD uses the same code system as TYPE_MAT_MAP
        self.MAT_SPRD_MAP = {
            '00': 'Type of material first ignited, other',
            '1': 'Flammable Gas',
            '10': 'Flammable gas, other',
            '11': 'Natural gas',
            '12': 'LP gas',
            '13': 'Anesthetic gas',
            '14': 'Acetylene',
            '15': 'Hydrogen',
            '2': 'Flammable, Combustible Liquid',
            '20': 'Flammable or combustible liquid, other',
            '21': 'Ether, pentane type flammable liquid',
            '22': 'JP-4 jet fuel & methyl ethyl ketone type flammable',
            '23': 'Gasoline',
            '24': 'Turpentine, butyl alcohol type flammable liquid',
            '25': 'Kerosene, No.1 and 2 fuel oil, diesel type',
            '26': 'Cottonseed oil, creosote oil type combustible',
            '27': 'Cooking oil, transformer or lubricating oil',
            '3': 'Volatile Solid or Chemical',
            '30': 'Volatile solid or chemical, other',
            '31': 'Fat, grease, butter, margarine, lard',
            '32': 'Petroleum jelly and non-food grease',
            '33': 'Polish, paraffin, wax',
            '34': 'Adhesive, resin, tar, glue, asphalt, pitch',
            '35': 'Paint, varnish - applied',
            '36': 'Combustible metal, included are magnesium',
            '37': 'Solid chemical, included are explosives',
            '38': 'Radioactive material',
            '4': 'Plastics',
            '41': 'Plastic',
            '5': 'Natural Product',
            '50': 'Natural product, other',
            '51': 'Rubber, excluding synthetic rubbers',
            '52': 'Cork',
            '53': 'Leather',
            '54': 'Hay, straw',
            '55': 'Grain, natural fiber,  (preprocess)',
            '56': 'Coal, coke, briquettes, peat',
            '57': 'Food, starch, excluding fat and grease (Code 31)',
            '58': 'Tobacco',
            '6': 'Wood or Paper - Processed',
            '60': 'Wood or paper, processed, other',
            '61': 'Wood chips, sawdust, shavings',
            '62': 'Round timber, including round posts, poles',
            '63': 'Sawn wood, including all finished lumber',
            '64': 'Plywood',
            '65': 'Fiberboard, particleboard, and hardboard',
            '66': 'Wood pulp',
            '67': 'Paper, including cellulose, waxed paper',
            '68': 'Cardboard',
            '7': 'Fabric, Textiles, Fur',
            '70': 'Fabric, textile, fur, other',
            '71': 'Fabric, fiber, cotton, blends, rayon, wool',
            '74': 'Fur, silk, other fabric.',
            '75': 'Wig',
            '76': 'Human hair',
            '77': 'Plastic coated fabric',
            '8': 'Material Compounded with Oil',
            '80': 'Material compounded with oil, other',
            '81': 'Linoleum',
            '82': 'Oilcloth',
            '86': 'Asphalt treated material',
            '9': 'Other Material',
            '99': 'Multiple types of material',
            'UU': 'Undetermined',
        }

        # ========================
        # New official NFIRS code tables (added per updated codelookup)
        # ========================

        self.ON_SITE_M1_MAP = {
            '0': 'On-site materials, other',
            '100': 'Foods, beverages, agriculture, other',
            '110': 'Food, other',
            '111': 'Baked goods',
            '112': 'Meat products, including poultry & fish',
            '114': 'Produce, fruit or vegetables',
            '118': 'Fat/cooking grease, including lard & animal fat',
            '120': 'Beverages, other',
            '130': 'Agriculture, other',
            '131': 'Trees, plants, flowers',
            '132': 'Feed, grain, seed',
            '133': 'Hay, straw',
            '135': 'Livestock',
            '136': 'Pets',
            '200': 'Personal & home products, other',
            '210': 'Fabrics, other',
            '211': 'Curtains, drapes',
            '212': 'Linens',
            '213': 'Bedding',
            '214': 'Cloth, yarn, dry goods',
            '220': 'Wearable products, other',
            '221': 'Clothes',
            '222': 'Footwear',
            '225': 'Perfumes, colognes, cosmetics',
            '226': 'Toiletries',
            '230': 'Accessories, other',
            '233': 'Purses, satchels, briefcases, wallets, belts',
            '240': 'Furnishings, other',
            '241': 'Furniture',
            '242': 'Beds, mattresses',
            '244': 'Houseware',
            '245': 'Glass, ceramics, china, pottery, stoneware',
            '300': 'Raw materials, other',
            '310': 'Wood, other',
            '311': 'Lumber, sawn wood',
            '312': 'Timber',
            '315': 'Sawdust, wood chips',
            '320': 'Fibers, other',
            '321': 'Cotton',
            '341': 'Ore',
            '342': 'Rubber',
            '343': 'Plastics',
            '344': 'Fiberglass',
            '345': 'Salt',
            '400': 'Paper products, rope, other',
            '410': 'Paper products, other',
            '411': 'Newspaper, magazines',
            '412': 'Books',
            '414': 'Paper - rolled',
            '415': 'Cardboard',
            '416': 'Packaged paper products, including stationary',
            '417': 'Paper records or reports',
            '500': 'Flammables, chemicals, plastics, other',
            '510': 'Flammables, combustible liquids, other',
            '511': 'Gasoline, diesel fuel',
            '512': 'Flammable liquid, not gasoline',
            '513': 'Combustible liquid, including heating oil',
            '514': 'Motor oil',
            '515': 'Heavy oils, grease, non-cooking related',
            '516': 'Asphalt',
            '517': 'Adhesive, resin, tar',
            '520': 'Flammable gas, other',
            '521': 'Natural gas',
            '522': 'LP gas, Butane, Propane',
            '523': 'Hydrogen gas',
            '531': 'Charcoal',
            '540': 'Chemicals, drugs, other',
            '541': 'Hazardous chemicals',
            '542': 'Non-hazardous chemicals',
            '600': 'Construction, machinery, metals, other',
            '610': 'Machinery, tools, other',
            '611': 'Industrial Machinery',
            '612': 'Machine parts',
            '613': 'Tools (power & hand tools)',
            '620': 'Construction supplies, other',
            '621': 'Hardware products',
            '622': 'Construction & home improvement products',
            '623': 'Pipes, fittings',
            '624': 'Stone-working materials',
            '626': 'Electrical: parts, supplies, equipment',
            '627': 'Insulation',
            '630': 'Floor & wall coverings, other',
            '631': 'Carpets, rugs',
            '632': 'Linoleum, tile',
            '635': 'Paint',
            '640': 'Metal products, other',
            '700': 'Appliances, electronics, medical, lab, other',
            '710': 'Appliances, electronics, other',
            '711': 'Appliances',
            '712': 'Electronic: parts, supplies, equipment',
            '713': 'Electronic media',
            '722': 'Medical supply',
            '810': 'Motor vehicles & parts, other',
            '811': 'Autos, trucks, buses, recreational vehicles',
            '813': 'Motor vehicle parts, not including tires',
            '814': 'Tires',
            '821': 'Boats, ships',
            '831': 'Planes, airplanes',
            '910': 'Containers, packing materials, other',
            '911': 'Bottles, barrels, boxes',
            '912': 'Packing material',
            '920': 'Previously owned products, other',
            '923': 'Used merchandise',
            '930': 'Ordnance, explosives, fireworks, other',
            '932': 'Ammunition',
            '933': 'Explosives',
            '934': 'Fireworks',
            '943': 'Art supply/artwork',
            '944': 'Sporting goods',
            '950': 'Mixed sales products, other',
            '951': 'Office supplies',
            '952': 'Restaurant supplies, not including food',
            '960': 'Discarded material, other',
            '962': 'Recyclable materials',
            '963': 'Trash, not recyclable',
        }

        self.MAT_STOR1_MAP = {
            '1': 'Bulk storage or warehousing',
            '2': 'Processing or manufacturing',
            '3': 'Packaged goods for sale',
            '4': 'Repair or service',
        }

        self.EQUIP_INV_MAP = {
            '0': 'Other equipment involved in ignition',
            '100': 'Heating, ventilating & air conditioning, other',
            '111': 'Air conditioner',
            '112': 'Heat pump',
            '113': 'Fan',
            '116': 'Dehumidifier',
            '117': 'Evaporative cooler, cooling tower.',
            '120': 'Fireplace, chimney, other',
            '121': 'Fireplace, masonry',
            '122': 'Fireplace, factory built',
            '123': 'Fireplace, insert/stove',
            '124': 'Stove, heating',
            '127': 'Chimney - metal, including stovepipe, flue',
            '131': 'Furnace, local heating unit, built-in',
            '132': 'Furnace, central heating unit',
            '133': 'Boiler (power, process, heating)',
            '141': 'Heater, excluding catalytic and oil-filled heaters',
            '142': 'Heater, catalytic',
            '143': 'Heater, oil filled',
            '144': 'Heat lamp',
            '145': 'Heat tape',
            '151': 'Water heater',
            '152': 'Steamline, heat pipe, hot air duct',
            '200': 'Electrical distribution, power transfer, other',
            '210': 'Electrical wiring, other',
            '211': 'Electrical power (utility) line',
            '212': 'Electrical service supply wires from utility',
            '213': 'Electric meter, meter box',
            '214': 'Wiring from meter box to circuit breaker',
            '215': 'Panelboard, switchboard, circuit breaker board',
            '216': 'Electrical branch circuit',
            '217': 'Outlet, receptacle',
            '218': 'Wall switch',
            '219': 'Ground fault interrupter, GFI',
            '221': 'Transformer, distribution type',
            '222': 'Overcurrent, disconnect equipment',
            '223': 'Transformer, low voltage',
            '224': 'Generator',
            '225': 'Inverter',
            '226': 'Uninterrupted power supply (UPS)',
            '227': 'Surge protector',
            '228': 'Battery charger, rectifier',
            '229': 'Battery',
            '230': 'Lamp, lighting, other',
            '231': 'Lamp - tabletop, floor, desk',
            '232': 'Lantern, flashlight',
            '233': 'Incandescent lighting fixture',
            '234': 'Fluorescent lighting fixture, ballast',
            '235': 'Halogen lighting fixture or lamp',
            '236': 'Sodium, mercury vapor lighting fixtures or lamps;',
            '237': 'Work light, trouble light',
            '238': 'Light bulb',
            '242': 'Decorative lights, line voltage',
            '244': 'Sign',
            '260': 'Cord, plug, other',
            '261': 'Power cord, plug - detachable from appliance',
            '262': 'Power cord, plug - permanently attached',
            '263': 'Extension cord',
            '300': 'Shop or industrial equipment, other',
            '310': 'Power tools, other',
            '311': 'Power saw',
            '314': 'Power cutting tool',
            '316': 'Power sander, grinder, buffer, polisher',
            '331': 'Welding torch.',
            '332': 'Cutting torch',
            '333': 'Burners',
            '334': 'Soldering equipment',
            '341': 'Air compressor',
            '344': 'Pump',
            '345': 'Wet/dry vacuum (shop vacuum)',
            '351': 'Heat treating equipment',
            '353': 'Industrial furnace, kiln',
            '354': 'Tarpot, tar kettle',
            '355': 'Casting, molding, forging equipment',
            '361': 'Conveyor',
            '373': 'Gas regulator',
            '374': 'Motor - separate',
            '375': 'Internal combustion engine (non-vehicular)',
            '377': 'Car washing equipment',
            '400': 'Commercial or medical equipment, other',
            '411': 'Dental, medical, or other powered bed or chair',
            '412': 'Dental equipment, other',
            '416': 'Oxygen administration equipment',
            '419': 'Therapeutic equipment',
            '423': 'TV monitor array',
            '433': 'Elevator or lift',
            '524': 'Lawn mower',
            '525': 'Lawn, landscape trimmer, edger',
            '532': 'Leaf blower',
            '535': 'Log splitter',
            '600': 'Kitchen & cooking equipment, other',
            '611': 'Blender, juicer, food processor, mixer',
            '631': 'Coffee maker or teapot',
            '632': 'Food warmer, hot plate',
            '633': 'Kettle',
            '635': 'Pressure cooker or canner',
            '636': 'Slow cooker',
            '637': 'Toaster, toaster oven, counter-top broiler',
            '639': 'Wok, frying pan, skillet',
            '642': 'Deep fryer',
            '643': 'Grill, hibachi, barbecue',
            '644': 'Microwave oven',
            '645': 'Oven, rotisserie',
            '646': 'Range, stove with/without oven, cooking surface',
            '651': 'Dishwasher',
            '652': 'Freezer when separate from refrigerator',
            '653': 'Garbage disposer',
            '654': 'Grease hood/duct exhaust fan',
            '655': 'Ice maker (separate from refrigerator)',
            '656': 'Refrigerator, refrigerator/freezer',
            '700': 'Electronic equipment, other',
            '710': 'Computer device, other',
            '711': 'Computer',
            '712': 'Computer storage device: external',
            '714': 'Computer monitor',
            '715': 'Computer printer',
            '716': 'Computer projection device, LCD panel',
            '722': 'Telephone or answering machine',
            '741': 'CD player (audio)',
            '743': 'Radio',
            '747': 'Speakers, audio - separate components',
            '748': 'Stereo equipment',
            '750': 'Video equipment, other',
            '751': 'Cable converter box',
            '753': 'Television',
            '755': 'Video game - electronic',
            '756': 'Camcorder, video camera',
            '800': 'Personal or household equipment, other',
            '811': 'Clothes dryer',
            '812': 'Trash compactor',
            '813': 'Washer/dryer combination (within one frame)',
            '814': 'Washing machine - clothes',
            '821': 'Hot tub, whirlspool, spa',
            '822': 'Swimming pool equipment',
            '834': 'Vacuum cleaner',
            '841': 'Comb, hair brush',
            '842': 'Curling iron',
            '844': 'Hair curler warmer',
            '845': 'Hair dryer',
            '850': 'Portable appliance designed to produce heat, other',
            '852': 'Blanket - electric',
            '855': 'Clothes iron',
            '863': 'Garage door opener',
            '868': 'Thermostat',
            '871': 'Ashtray',
            '872': 'Charcoal lighter',
            '873': 'Cigarette lighter, pipe lighter',
            '881': 'Model vehicles.',
            '882': 'Toy, powered',
            '883': 'Woodburning kit',
            '891': 'Clock',
        }

    def map_code(self, code_map: Dict[str, str], value: Any, fallback: str = None) -> str:
        """Map code to description"""
        v = None if value is None or (isinstance(value, float) and np.isnan(value)) else str(value).strip()
        if not v:
            return fallback
        return code_map.get(v, fallback)

    def _get_season(self, month: Any) -> str:
        """获取季节"""
        try:
            m = int(month)
            return self.SEASON_MAP.get(m, 'Unknown')
        except (ValueError, TypeError):
            return 'Unknown'

    def _get_time_period(self, hour: Any) -> str:
        """获取时间段"""
        try:
            h = int(hour)
            if 6 <= h < 12:
                return 'morning'
            elif 12 <= h < 18:
                return 'afternoon'
            elif 18 <= h < 22:
                return 'evening'
            else:  # 22-23 or 0-5
                return 'night'
        except (ValueError, TypeError):
            return 'Unknown'

    def _clean_str(self, v: Any) -> str:
        """Clean string value"""
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 'Unknown'
        s = str(v).strip()
        return s if s else 'Unknown'

    def _is_missing(self, v: Any) -> bool:
        """Check if value is missing"""
        if v is None:
            return True
        s = str(v).strip()
        return s == "" or s.lower() in {"unknown", "not_given", "not given", "not given_", "none", "nan", "uu_", "uu", "undetermined"}

    def _fmt_money(self, v: Any) -> str:
        """Format monetary value"""
        try:
            return f"${float(v):,.0f}"
        except Exception:
            return str(v) if v is not None else "Unknown"

    def _fmt_pct(self, v: Any) -> str:
        """Format percentage value"""
        try:
            return f"{float(v)}"
        except Exception:
            return str(v) if v is not None else "Unknown"

    def create_fire_prompt(self, incident_data: FireIncidentData) -> str:
        """Create fire incident prompt based on data (matching new template format)."""

        bi = incident_data.building_info or {}
        wi = incident_data.weather_info or {}
        fi = incident_data.fire_info or {}
        cc = incident_data.community_context or {}

        lines = []

        # Header
        lines.append(f"<incident_key: {incident_data.incident_key}>")
        lines.append("Task Information:")
        lines.append("You are a helpful assistant designed to forecast fire severity for a specific building. Your primary task is to predict the probability of fire spread using the following classification options:")
        lines.append("[Fire confined in the room, Fire confined in the floor, Fire beyond the floor].")
        lines.append("Fire severity can be categorized as confined in the room (lowest, localized), confined in the floor (moderate, same floor spread), or beyond floor (high, multi-floor or building spread).")
        lines.append("")

        # Building Information
        lines.append("% Basic Building Information:")
        # if not (self._is_missing(bi.get('state')) or self._is_missing(bi.get('zip_code'))):
            # lines.append(f"Location: The building is located in <{bi.get('state')}>, zip code <{bi.get('zip_code')}>.")

        # 建筑占用和结构信息 - 分别处理每个部分
        if not self._is_missing(bi.get('occupant_type_desc')):
            lines.append(f"Occupancy: It is classified as a <{bi.get('occupant_type_desc')}> property.")
        # if not self._is_missing(bi.get('num_units')):
            # lines.append(f"Units: The building has <{bi.get('num_units')}> units.")
        if not (self._is_missing(bi.get('stories_above')) or self._is_missing(bi.get('stories_below'))):
            lines.append(f"Structure: The building has <{bi.get('stories_above')}> stories above ground and <{bi.get('stories_below')}> story below ground.")
        # 建筑构造信息 - 分别处理
        if not self._is_missing(bi.get('build_year')) and bi.get('build_year') != 'Unknown':
            lines.append(f"Construction Year: Constructed around the year <{bi.get('build_year')}>.")
        # if not self._is_missing(bi.get('build_material')) and bi.get('build_material') != 'Unknown':
        #     lines.append(f"Construction Material: The building is primarily made of <{bi.get('build_material')}>.")
        # 现场材料信息 - 分别处理
        if not self._is_missing(bi.get('on_site_material_major')):
            lines.append(f"On-site Material: The most significant on-site material is <{bi.get('on_site_material_major')}>.")
        if not self._is_missing(bi.get('material_storage_use')):
            lines.append(f"Storage Use: The material storage use is <{bi.get('material_storage_use')}>.")
        lines.append("")

        # Data correction criteria
        lines.append("%Data correction criteria:")
        lines.append("Please apply the following NFIRS data rules:")
        lines.append("If a fire starts on the roof or an exterior wall, or if the structure is a one-room building, the outcome should be classified as 'Fire confined in the building'.")

        # Incident Conditions
        lines.append("%Incident Conditions:")
        if not (self._is_missing(wi.get('temperature')) or self._is_missing(wi.get('humidity'))):
            lines.append(f"Weather: At the time of the incident, the ambient temperature was <{wi.get('temperature')}°C> with <{wi.get('humidity')}>% relative humidity.")

        # 火灾起源信息 - 分别处理
        # if not (self._is_missing(fi.get('date')) or self._is_missing(fi.get('time'))):
        #     lines.append(f"Fire Time: The fire started on <{fi.get('date')}> at approximately <{fi.get('time')}>.")
        if not (self._is_missing(fi.get('date')) or self._is_missing(fi.get('time'))):
            lines.append(f"Fire Time: The fire occurred in <{fi.get('date')}> during the <{fi.get('time')}>.")
        if not self._is_missing(fi.get('origin_desc')):
            lines.append(f"Fire Location: The fire originated in the <{fi.get('origin_desc')}>.")
        if not self._is_missing(fi.get('fire_origin_floor')):
            lines.append(f"Floor: The fire started on floor <{fi.get('fire_origin_floor')}>.")

        # 点火源信息 - 分别处理
        if not self._is_missing(fi.get('heat_source_desc')):
            lines.append(f"Heat Source: The heat source was <{fi.get('heat_source_desc')}>.")
        if not self._is_missing(fi.get('equipment_involved_desc')):
            lines.append(f"Equipment: The equipment involved was <{fi.get('equipment_involved_desc')}>.")
        if not self._is_missing(fi.get('ignited_material_desc')):
            lines.append(f"First Ignited: The first item to ignite was <{fi.get('ignited_material_desc')}>.")
        if not self._is_missing(fi.get('type_material_desc')):
            lines.append(f"Material Type: The material type is <{fi.get('type_material_desc')}>.")

        # 火灾扩散信息 - 分别处理
        if not self._is_missing(fi.get('item_contributing_spread_desc')):
            lines.append(f"Spread Item: The primary item contributing to the fire's spread was <{fi.get('item_contributing_spread_desc')}>.")
        if not self._is_missing(fi.get('material_contributing_spread_desc')):
            lines.append(f"Spread Material: The material contributing to spread was <{fi.get('material_contributing_spread_desc')}>.")

        # 安全系统信息 - 分别处理
        if not self._is_missing(fi.get('detector_status_desc')):
            lines.append(f"Detector: The detector status was <{fi.get('detector_status_desc')}>.")
        if not self._is_missing(fi.get('ase_status_desc')):
            lines.append(f"Suppression System: The automatic extinguishing system (AES) status was <{fi.get('ase_status_desc')}>.")

        # 影响因素 - 分别处理
        if not self._is_missing(fi.get('human_factor_primary_desc')):
            lines.append(f"Human Factor: The primary human factor contributing to ignition was <{fi.get('human_factor_primary_desc')}>.")
        if not self._is_missing(fi.get('factor_ignition_primary_desc')):
            lines.append(f"Physical Factor: The main contributing physical factor was <{fi.get('factor_ignition_primary_desc')}>.")
        if not self._is_missing(fi.get('cause_ignition_desc')):
            lines.append(f"Cause: The reported cause of ignition is <{fi.get('cause_ignition_desc')}>.")

        # 灭火努力 - 分别处理
        if not self._is_missing(fi.get('response_time')):
            lines.append(f"Response Time: Fire department response time was approximately <{fi.get('response_time')} min>.")
        if not self._is_missing(fi.get('firefighter_action_primary_desc')):
            lines.append(f"Firefighter Action: The primary suppression action taken by firefighters was <{fi.get('firefighter_action_primary_desc')}>.")
        # if not self._is_missing(fi.get('suppression_factor_primary_desc')):
            # lines.append(f"Suppression Factor: The main factor that hindered suppression efforts was <{fi.get('suppression_factor_primary_desc')}>.")
        lines.append("")

        # Community Context
        lines.append("% Community Context:")
        if not self._is_missing(cc.get('median_income_level')):
            income_val = f" ({self._fmt_money(cc.get('median_income'))})" if not self._is_missing(cc.get('median_income')) else ""
            lines.append(f"This building is situated in a community where the median income level is <{cc.get('median_income_level')}{income_val}>.")
        if not self._is_missing(cc.get('median_rent_level')):
            rent_val = f" ({self._fmt_money(cc.get('median_rent'))})" if not self._is_missing(cc.get('median_rent')) else ""
            lines.append(f"The median monthly rent level is <{cc.get('median_rent_level')}{rent_val}>.")
        if not self._is_missing(cc.get('housing_occupancy_level')):
            occ_val = f" ({self._fmt_pct(cc.get('housing_occupancy'))}%)" if not self._is_missing(cc.get('housing_occupancy')) else ""
            lines.append(f"Housing occupancy level is <{cc.get('housing_occupancy_level')}{occ_val}>.")
        if not self._is_missing(cc.get('bachelor_degree_level')):
            deg_val = f" ({self._fmt_pct(cc.get('bachelor_degree'))}%)" if not self._is_missing(cc.get('bachelor_degree')) else ""
            lines.append(f"Population with bachelor's degree or higher level is <{cc.get('bachelor_degree_level')}{deg_val}>. ")
        if not self._is_missing(cc.get('elderly_population_level')):
            eld_val = f" ({self._fmt_pct(cc.get('elderly_population'))}%)" if not self._is_missing(cc.get('elderly_population')) else ""
            lines.append(f"Elderly population (62+) level is <{cc.get('elderly_population_level')}{eld_val}>. ")
        if not self._is_missing(cc.get('black_population_level')):
            bp_raw = cc.get('black_population')
            black_val = f" ({self._fmt_pct(round(float(bp_raw) * 100, 1))}%)" if not self._is_missing(bp_raw) else ""
            lines.append(f"Black or African American population level is <{cc.get('black_population_level')}{black_val}>.")
        lines.append("")

        # Task prompt
        lines.append("% Task Prompt:")
        lines.append("Based on all the information provided, predict the fire spread status from the given options.")
        lines.append("Answer: Fire spread status: <")

        prompt = "\n".join(lines)
        return prompt

    def create_quantile_based_labels(self, data_series: pd.Series, field_name: str) -> pd.Series:
        """基于分位数创建标签"""
        # 计算20%、40%、60%、80%分位数
        threshold_20 = data_series.quantile(0.2)
        threshold_40 = data_series.quantile(0.4)
        threshold_60 = data_series.quantile(0.6)
        threshold_80 = data_series.quantile(0.8)

        # 创建标签
        labels = pd.cut(
            data_series,
            bins=[-np.inf, threshold_20, threshold_40, threshold_60, threshold_80, np.inf],
            labels=['national lowest', 'below national average', 'national average', 'above national average', 'national highest'],
            include_lowest=True
        )

        # 转换为字符串类型
        return labels.astype(str)

    def create_distribution_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """基于实际数据的分位数创建标签，确保标签的合理性"""

        # 字段映射：标签字段名 -> 实际数据字段名
        field_mapping = {
            'median_income_level': 'median_income_list',      # 收入水平
            'median_rent_level': 'median_rent_list',          # 租金水平
            'housing_occupancy_level': 'Pct_HOU_Occupied_units_list',  # 住房占用率水平
            'bachelor_degree_level': 'Pct_EDU_Bachelor_or_higher_list',    # 学士学位水平
            'elderly_population_level': 'Pct_SA_62_and_over_list', # 老年人口水平
            'black_population_level': 'Black Alone',          # 黑人人口水平
            'employment_rate_level': 'Pct_EMP_Pop_16_and_over_in_labor_force_Civilian_labor_force_Employed_list'  # 就业率水平
        }

        # 为每个字段创建基于分位数的标签
        for label_field, data_field in field_mapping.items():
            if data_field in df.columns:
                # 基于实际数据的分位数创建标签
                df[label_field] = self.create_quantile_based_labels(
                    df[data_field],
                    label_field
                )

                # 验证标签分布的合理性
                self._validate_label_distribution(df, label_field, data_field)
            else:
                # 如果数据字段不存在，使用合理的默认值
                logging.warning(f"Field {data_field} not found; using default label for {label_field}")
                df[label_field] = 'medium'

        return df

    def _validate_label_distribution(self, df: pd.DataFrame, label_field: str, data_field: str):
        """验证标签分布的合理性"""
        label_counts = df[label_field].value_counts()
        total_count = len(df)

        # 检查每个标签的比例是否在合理范围内（期望约20%，允许10%-30%）
        for label in ['national lowest', 'below national average', 'national average', 'above national average', 'national highest']:
            if label in label_counts:
                percentage = (label_counts[label] / total_count) * 100
                if percentage < 10 or percentage > 30:
                    logging.debug(f"{label_field}: label {label} distribution {percentage:.1f}% may be imbalanced")

        # 验证标签与原始数据的逻辑一致性
        self._validate_label_logic(df, label_field, data_field)

        logging.debug(f"{label_field} labels created, distribution: {dict(label_counts)}")

    def _validate_label_logic(self, df: pd.DataFrame, label_field: str, data_field: str):
        """验证标签与原始数据的逻辑一致性"""
        # 获取每个标签组的统计信息
        label_stats = df.groupby(label_field)[data_field].agg(['mean', 'min', 'max'])

        # 验证标签的数值顺序是否合理
        if 'lowest' in label_stats.index and 'highest' in label_stats.index:
            low_mean = label_stats.loc['lowest', 'mean']
            high_mean = label_stats.loc['highest', 'mean']

            if low_mean >= high_mean:
                logging.warning(f"{label_field} label logic issue: lowest mean ({low_mean:.2f}) >= highest mean ({high_mean:.2f})")
            else:
                logging.debug(f"{label_field} label logic OK: lowest({low_mean:.2f}) < highest({high_mean:.2f})")

    def convert_dataframe_to_incident_data(self, df: pd.DataFrame) -> List[FireIncidentData]:
        """将DataFrame转换为FireIncidentData对象列表（适配ny_data_balanced.csv列名）
        注意：所有 code 列的读取均直接使用不带 _new 后缀的原始 NFIRS 列。"""
        incident_data_list = []

        for _, row in df.iterrows():
            # 构建建筑信息
            building_info = {
                'state': row.get('STATE', 'Unknown'),
                'zip_code': str(row.get('ZIP5', 'Unknown')),
                'occupant_type': row.get('PROP_USE', 'Unknown'),
                'occupant_type_desc': self._get_desc(row, 'PROP_USE', self.PROP_USE_MAP),
                'stories_above': row.get('BLDG_ABOVE', 'Unknown'),
                'stories_below': row.get('BLDG_BELOW', 'Unknown'),
                'num_units': row.get('NUM_UNIT', 'Unknown'),
                'square_footage': row.get('TOT_SQ_FT', 'Unknown'),
                'build_year': 'Unknown',  # 纽约数据中没有具体建筑年份信息
                # STRUC_TYPE 数据中没有 _new / _desc 列，只能沿用手写 Map 查表
                'structure_type': self.map_code(self.STRUC_TYPE_MAP, row.get('STRUC_TYPE'), 'Unknown'),
                # on_site_material_major / material_storage_use：使用官方 ON_SITE_M1_MAP / MAT_STOR1_MAP 查表
                # （优先读取 *_desc 列，缺失时回退到裸列查表，再查不到就是 'Unknown'）
                'on_site_material_major': self._get_desc(row, 'ON_SITE_M1', self.ON_SITE_M1_MAP),
                'material_storage_use': self._get_desc(row, 'MAT_STOR1', self.MAT_STOR1_MAP),
            }

            # 构建天气信息
            weather_info = {
                'temperature': row.get('temp', 'Unknown'),
                'humidity': row.get('rhum', 'Unknown')
            }

            # 构建火灾信息
            fire_info = {
                'origin': row.get('AREA_ORIG', 'Unknown'),
                # origin_desc: 经与 balanced.csv + codelookup14.txt 核对，FIRE_ORIG 字段
                # 实际存放的是"起火楼层/楼层编号"（取值如 1,2,3,-1,-2,999,800...），并不是
                # NFIRS 区域起源编码（codelookup14.txt 中根本没有 FIRE_ORIG 这个 fieldid）。
                # 真正的"起火区域"编码字段是 AREA_ORIG，因此这里直接用 AREA_ORIG 取描述，
                # 不再经过 FIRE_ORIG_MAP（否则 FIRE_ORIG=1 的多数行会被误判为 "Corridor, mall" 等）。
                'origin_desc': self._get_desc(row, 'AREA_ORIG', self.AREA_ORIG_MAP),
                'time': self._get_time_period(row.get('accident_hour')) if pd.notna(row.get('accident_hour')) else 'Unknown',
                'date': self._get_season(row.get('accident_month')) if pd.notna(row.get('accident_month')) else 'Unknown',
                'heat_source': row.get('HEAT_SOURC', 'Unknown'),
                # code 列与 desc 列均使用裸列 HEAT_SOURC（原 HEAT_SOURCE_new 兜底已去除）；
                # 但实际 desc 列名是 HEAT_SOURC_desc，与 base_col 同名，可省略 desc_col，这里显式写出以保持清晰
                'heat_source_desc': self._get_desc(row, 'HEAT_SOURC', self.HEAT_SOURC_MAP, desc_col='HEAT_SOURC_desc'),
                'ignited_material': row.get('FIRST_IGN', 'Unknown'),
                'ignited_material_desc': self._get_desc(row, 'FIRST_IGN', self.FIRST_IGN_MAP),
                'type_material_desc': self._get_desc(row, 'TYPE_MAT', self.TYPE_MAT_MAP),
                'floor': row.get('FIRE_ORIG', 'Unknown'),
                'fire_origin_floor': row.get('FIRE_ORIG', 'Unknown'),
                # DETECTOR / AES_PRES：使用官方 DETECTOR_MAP / AES_PRES_MAP 查表
                'detector_status': self._get_desc(row, 'DETECTOR', self.DETECTOR_MAP),
                'detector_status_desc': self._get_desc(row, 'DETECTOR', self.DETECTOR_MAP),
                'ase_status': self._get_desc(row, 'AES_PRES', self.AES_PRES_MAP),
                'ase_status_desc': self._get_desc(row, 'AES_PRES', self.AES_PRES_MAP),
                'item_contributing_spread_desc': self._get_desc(row, 'ITEM_SPRD', self.ITEM_SPRD_MAP),
                # MAT_SPRD_desc 列实测是错位的编码而非文本描述（数据bug），跳过，直接用 MAT_SPRD_MAP 查表（裸列 MAT_SPRD）
                'material_contributing_spread_desc': self.map_code(
                    self.MAT_SPRD_MAP, row.get('MAT_SPRD', 'Unknown'), 'Unknown'
                ),
                'response_time': row.get('response_time', 'Unknown'),
                'human_factor_primary_desc': self._get_desc(row, 'hum_1', self.HUM1_MAP),
                # 修正：实际列名是 FACT_IGN_1（下划线在1之前），不是 FACT_IGN1；
                # desc 列名是 FACT_IGN_1_desc（与 base_col 同名，可省略 desc_col，这里显式写出以保持清晰）
                'factor_ignition_primary_desc': self._get_desc(row, 'FACT_IGN_1', self.FACT_IGN1_MAP, desc_col='FACT_IGN_1_desc'),
                'firefighter_action_primary_desc': self._get_desc(row, 'ACT_TAK1', self.ACT_TAK1_MAP),
                # 'suppression_factor_primary_desc': self._get_desc(row, 'SUP_FAC_1', self.SUP_FAC1_MAP),
                'cause_ignition_desc': self._get_desc(row, 'CAUSE_IGN', self.CAUSE_IGN_MAP),
                # equipment_involved_desc：使用官方 EQUIP_INV_MAP 查表
                'equipment_involved_desc': self._get_desc(row, 'EQUIP_INV', self.EQUIP_INV_MAP),
            }

            # 构建社区上下文 - 使用分位数标签
            community_context = {
                # 分位标签（保留）
                'median_income_level': row.get('median_income_level', 'Unknown'),
                'median_rent_level': row.get('median_rent_level', 'Unknown'),
                'housing_occupancy_level': row.get('housing_occupancy_level', 'Unknown'),
                'bachelor_degree_level': row.get('bachelor_degree_level', 'Unknown'),
                'elderly_population_level': row.get('elderly_population_level', 'Unknown'),
                'black_population_level': row.get('black_population_level', 'Unknown'),
                # 原始数值（用于格式化输出）
                'median_income': row.get('median_income_list', 'Unknown'),
                'median_rent': row.get('median_rent_list', 'Unknown'),
                'housing_occupancy': row.get('Pct_HOU_Occupied_units_list', 'Unknown'),
                'bachelor_degree': row.get('Pct_EDU_Bachelor_or_higher_list', 'Unknown'),
                'elderly_population': row.get('Pct_SA_62_and_over_list', 'Unknown'),
                'employment_rate': row.get('Pct_EMP_Pop_16_and_over_in_labor_force_Civilian_labor_force_Employed_list', 'Unknown'),
                'black_population': row.get('Black Alone', row.get('Minority populations', 'Unknown')),
                'coal_wood_heating': row.get('coal_or_wood', 'Unknown')
            }

            # 创建FireIncidentData对象
            incident_data = FireIncidentData(
                incident_key=row.get('INCIDENT_KEY', f'incident_{len(incident_data_list)}'),
                building_info=building_info,
                weather_info=weather_info,
                fire_info=fire_info,
                community_context=community_context,
                fire_spread_label=row.get('fire_spread_label', 2)  # 使用映射后的标签(2,3,4)
            )

            incident_data_list.append(incident_data)

        return incident_data_list

    def create_fire_spread_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建火灾扩散标签 - 保持原始分布"""
        # 使用实际的FIRE_SPRD字段，保持原始分布
        if 'FIRE_SPRD' in df.columns:
            # 映射到我们的标签系统（3类：2,3,4，其中4类包含原来的4和5）
            # 2: fire within the room, 3: within the floor, 4: beyond the floor (包含原来的4和5)
            fire_spread_mapping = {
                1: 2,  # 房间内
                2: 2,  # 房间内
                3: 3,  # 楼层内
                4: 4,  # 建筑内
                5: 4,  # 建筑外 -> 合并到4类
                45: 4  # 建筑内外合并值 -> 映射到4类
            }
            df['fire_spread_label'] = df['FIRE_SPRD'].map(fire_spread_mapping).fillna(2)
        else:
            # 如果没有FIRE_SPRD字段，创建模拟标签（3类分布）
            np.random.seed(42)
            df['fire_spread_label'] = np.random.choice([2, 3, 4], size=len(df), p=[0.5, 0.3, 0.2])

        return df

    def generate_prompt_from_csv(self, csv_path: str) -> str:
        """从CSV文件生成示例prompt"""
        import pandas as pd

        # 读取CSV文件
        df = pd.read_csv(csv_path)

        # 创建分位数标签
        df = self.create_distribution_labels(df)

        # 创建火灾扩散标签
        df = self.create_fire_spread_labels(df)

        # 转换为FireIncidentData对象
        incidents = self.convert_dataframe_to_incident_data(df)

        if not incidents:
            raise ValueError("没有可用的事件数据，无法生成示例prompt")

        # 生成prompt
        prompt = self.create_fire_prompt(incidents[0])
        return prompt

    def generate_multiple_prompts_from_csv(self, csv_path: str, num_examples: int = 10) -> List[str]:
        """从CSV文件生成多条示例prompt

        Args:
            csv_path: CSV文件路径
            num_examples: 要生成的示例数量，默认10条

        Returns:
            包含多个prompt字符串的列表
        """
        import pandas as pd

        # 读取CSV文件
        df = pd.read_csv(csv_path)

        # 创建分位数标签
        df = self.create_distribution_labels(df)

        # 创建火灾扩散标签
        df = self.create_fire_spread_labels(df)

        # 转换为FireIncidentData对象
        incidents = self.convert_dataframe_to_incident_data(df)

        if not incidents:
            raise ValueError("没有可用的事件数据，无法生成示例prompt")

        # 生成指定数量的prompts
        actual_num = min(num_examples, len(incidents))
        prompts = []
        for i in range(actual_num):
            prompt = self.create_fire_prompt(incidents[i])
            prompts.append(prompt)

        return prompts


# 为了向后兼容，提供一些便捷函数
def create_fire_prompt(incident_data: FireIncidentData) -> str:
    """便捷函数：创建火灾事件prompt"""
    generator = FirePromptGenerator()
    return generator.create_fire_prompt(incident_data)


def convert_dataframe_to_incident_data(df: pd.DataFrame) -> List[FireIncidentData]:
    """便捷函数：将DataFrame转换为FireIncidentData对象列表"""
    generator = FirePromptGenerator()
    return generator.convert_dataframe_to_incident_data(df)


def create_quantile_based_labels(data_series: pd.Series, field_name: str) -> pd.Series:
    """便捷函数：基于分位数创建标签"""
    generator = FirePromptGenerator()
    return generator.create_quantile_based_labels(data_series, field_name)


if __name__ == "__main__":
    # 测试代码
    csv_path = "./data/processed_data/new_cleaned_combined_260704/balanced.csv"
    generator = FirePromptGenerator()

    try:
        # 调用新方法生成10条示例
        prompts = generator.generate_multiple_prompts_from_csv(csv_path, num_examples=10)

        print(f"\n{'='*80}")
        print(f"Successfully generate {len(prompts)} Prompts")
        print(f"{'='*80}\n")

        for i, prompt in enumerate(prompts, 1):
            print(f"\n{'='*80}")
            print(f"Example {i}/{len(prompts)}")
            print(f"{'='*80}")
            print(prompt)
            print(f"{'='*80}\n")

    except FileNotFoundError:
        print(f"❌ 未找到文件: {csv_path}")
    except Exception as e:
        print(f"❌ 生成示例prompt失败: {e}")
        import traceback
        traceback.print_exc()
