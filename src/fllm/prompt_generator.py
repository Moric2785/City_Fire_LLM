

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
    
    def __init__(self):
        """Initialize prompt generator"""
        # Fire spread class mapping (3 classes: 2,3,4, where class 4 includes original 4 and 5)
        self.fire_spread_classes = {
            2: "fire within the room",
            3: "within the floor", 
            4: "beyond the floor"
        }
        
        # Initialize encoding mappings
        self._init_code_mappings()
    
    def _init_code_mappings(self):
        """Initialize various encoding mappings"""
        self.AREA_ORIG_MAP = {
            '24': 'Sleeping room or bedroom for 1-2 persons',
            '21': 'Kitchen or cooking area',
            '14': 'Lobby or waiting area',
            '72': 'Chimney - exterior',
            '26': 'Bathroom or washroom',
            '20': 'Living area, other',
            '25': 'Laundry area',
            '47': 'Garage or carport',
            '76': 'Exterior wall surface'
        }
        
        self.FIRE_ORIG_MAP = {
            1.0: 'Kitchen',
            2.0: 'Bedroom', 
            3.0: 'Living room',
            4.0: 'Laundry area',
            5.0: 'Electrical',
            6.0: 'Heating system',
            7.0: 'Bathroom',
            8.0: 'Garage',
            9.0: 'Attic',
            10.0: 'Basement',
            11.0: 'Other'
        }
        
        self.HEAT_SOURC_MAP = {
            '1': 'Heat from Powered Equipment',
            '2': 'Radiated, Conducted Heat from Operating Equipment',
            '3': 'Electrical Arcing',
            '4': 'Heat from Hot or Smoldering Object',
            '5': 'Explosives, Fireworks',
            '7': 'Chemical, Natural Heat Sources (e.g., Spontaneous Combustion, Lightning)',
            '8': 'Heat Spread From Another Fire',
            'UU_': 'Undetermined',
            'UU': 'Undetermined'
        }
        
        self.FIRST_IGN_MAP = {
            '1': 'Structural Component or Finish (e.g., exterior siding, interior wall covering)',
            '2': 'Furniture and Utensils (e.g., upholstered sofa, cabinetry)',
            '3': 'Soft Goods and Wearing Apparel (e.g., mattress, bedding, clothing)',
            '4': 'Adornment, Recreational Material, Signs (e.g., Christmas tree, decoration)',
            '5': 'Storage Supplies (e.g., box, carton, pallet)',
            '6': 'General Flammable Liquids, Gases, and Chemicals',
            '7': 'Organic Materials (e.g., agricultural crops, vegetation)',
            '8': 'Other Material Compounded with Oil (e.g., linoleum, oilcloth)',
            '9': 'General Materials (e.g., books, rubbish, oily rags)',
            'UU_': 'Undetermined',
            'UU': 'Undetermined'
        }
        
        # self.TYPE_MAT_MAP = {
        #     '71': 'Plastic',
        #     '41': 'Wood, bark - hard, soft',
        #     '63': 'Natural fiber (textile)'
        # }

        self.TYPE_MAT_MAP = {
            '00': 'Type of material first ignited, other',
            # Flammable Gas
            '1': 'Flammable Gas',
            '10': 'Flammable gas, other',
            '11': 'Natural gas',
            '12': 'LP gas',
            '13': 'Anesthetic gas',
            '14': 'Acetylene',
            '15': 'Hydrogen',
            # Flammable, Combustible Liquid
            '2': 'Flammable, Combustible Liquid',
            '20': 'Flammable or combustible liquid, other',
            '21': 'Ether, pentane type flammable liquid',
            '22': 'JP-4 jet fuel & methyl ethyl ketone type flammable',
            '23': 'Gasoline',
            '24': 'Turpentine, butyl alcohol type flammable liquid',
            '25': 'Kerosene, No.1 and 2 fuel oil, diesel type',
            '26': 'Cottonseed oil, creosote oil type combustible',
            '27': 'Cooking oil, transformer or lubricating oil',
            '28': 'Ethanol',
            # Volatile Solid or Chemical
            '3': 'Volatile Solid or Chemical',
            '30': 'Volatile solid or chemical, other',
            '31': 'Fat, grease, butter, margarine, lard',
            '32': 'Petroleum jelly and non-food grease',
            '33': 'Polish, paraffin, wax',
            '34': 'Adhesive, resin, tar, glue, asphalt, pitch',
            '35': 'Paint, varnish - applied',
            '36': 'Combustible metal (e.g., magnesium)',
            '37': 'Solid chemical (e.g., explosives)',
            '38': 'Radioactive material',
            # Plastics
            '4': 'Plastics',
            '41': 'Plastic',
            # Natural Product
            '5': 'Natural Product',
            '50': 'Natural product, other',
            '51': 'Rubber (excluding synthetic rubbers)',
            '52': 'Cork',
            '53': 'Leather',
            '54': 'Hay, straw',
            '55': 'Grain, natural fiber (preprocess)',
            '56': 'Coal, coke, briquettes, peat',
            '57': 'Food, starch (excluding fat and grease)',
            '58': 'Tobacco',
            # Wood or Paper - Processed
            '6': 'Wood or Paper - Processed',
            '60': 'Wood or paper, processed, other',
            '61': 'Wood chips, sawdust, shavings',
            '62': 'Round timber (posts, poles)',
            '63': 'Sawn wood (finished lumber)',
            '64': 'Plywood',
            '65': 'Fiberboard, particleboard, hardboard',
            '66': 'Wood pulp',
            '67': 'Paper (cellulose, waxed paper)',
            '68': 'Cardboard',
            # Fabric, Textiles, Fur
            '7': 'Fabric, Textiles, Fur',
            '70': 'Fabric, textile, fur, other',
            '71': 'Fabric, fiber (cotton, blends, rayon, wool)',
            '74': 'Fur, silk, other fabric',
            '75': 'Wig',
            '76': 'Human hair',
            '77': 'Plastic coated fabric',
            # Material Compounded with Oil
            '8': 'Material Compounded with Oil',
            '80': 'Material compounded with oil, other',
            '81': 'Linoleum',
            '82': 'Oilcloth',
            '86': 'Asphalt treated material',
            # Other Material
            '9': 'Other Material',
            '99': 'Multiple types of material',
            'UU': 'Undetermined'
        }
        
        self.ACT_TAK1_MAP = {
            '11': 'Extinguishment',
            '10': 'Action taken, other',
            '12': 'Salvage & Overhaul',
            '86': 'Investigate',
            '87': 'Provide information'
        }
        
        self.FACT_IGN1_MAP = {
            '12': 'Mechanical failure or malfunction, other',
            '11': 'Worn out',
            '30': 'Design, manufacturing, installation deficiency, other'
        }
        
        self.SUP_FAC1_MAP = {
            'NNN': 'No factor suppressed the fire',
            '100': 'Fire protection system, other',
            '411': 'Fire door',
            '112': 'Sprinkler system, wet pipe'
        }
        
        self.CAUSE_IGN_MAP = {
            '1': 'Intentional',
            '2': 'Unintentional',
            '3': 'Failure of equipment or heat source',
            '4': 'Act of nature',
            '5': 'Exposure',
            'U': 'Cause under investigation'
        }
        
        self.HUM1_MAP = {
            'N': 'None',
            '1': 'Asleep',
            '2': 'Unattended or unsupervised person',
            '3': 'Possibly impaired by alcohol or drugs',
            '4': 'Physically disabled or handicapped',
            '5': 'Mentally disabled or handicapped',
            '6': 'Age was a factor',
            '7': 'Unfamiliar with material or equipment'
        }
        
        self.PROP_USE_MAP = {
            1: 'Assembly (e.g., theater, restaurant, church)',
            2: 'Educational',
            3: 'Institutional (e.g., hospital, school, correctional facility)',
            4: 'Residential (e.g., 1- or 2-family dwelling, multifamily dwelling)',
            5: 'Mercantile, Business',
            6: 'Basic Industry, Utility, Defense',
            7: 'Manufacturing, Processing',
            8: 'Storage',
            9: 'Outside or Special Property (e.g., open land, construction site)'
        }
        # ========================
        self.STRUC_TYPE_MAP = {
            '0': 'Structure type, other',
            '1': 'Enclosed building',
            '2': 'Fixed portable or mobile structure',
            '3': 'Open structure',
            '4': 'Air supported structure',
            '5': 'Tent',
            '6': 'Open platform',
            '7': 'Underground structure work areas',
            '8': 'Connective structure'
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
            '00': 'Item First Ignited, Other',
            '1': 'Structural Component, Finish',
            '10': 'Structural component or finish, other',
            '11': 'Exterior roof covering or finish',
            '12': 'Exterior wall covering or finish',
            '13': 'Exterior trim, including doors',
            '14': 'Floor covering or rug/carpet/mat',
            '15': 'Interior wall covering excluding drapes, etc.',
            '16': 'Interior ceiling cover or finish',
            '17': 'Structural member or framing',
            '18': 'Insulation within structural area',
            '2': 'Furniture, Utensils',
            '20': 'Furniture, utensils, other',
            '21': 'Upholstered sofa, chair, vehicle seats',
            '22': 'Non-upholstered chair, bench',
            '23': 'Cabinetry (including built-in)',
            '24': 'Ironing board',
            '25': 'Appliance housing or casing',
            '26': 'Household utensils',
            '3': 'Soft Goods, Wearing Apparel',
            '30': 'Soft goods, wearing apparel, other',
            '31': 'Mattress, pillow',
            '32': 'Bedding; blanket, sheet, comforter',
            '33': 'Linen; other than bedding',
            '34': 'Wearing apparel not on a person',
            '35': 'Wearing apparel on a person',
            '36': 'Curtains, blinds, drapery, tapestry',
            '37': 'Goods not made up, including fabrics & yard goods',
            '38': 'Luggage',
            '4': 'Adornment, Recreational Material, Signs',
            '40': 'Adornment, recreational material, signs, other',
            '41': 'Christmas tree',
            '42': 'Decoration',
            '43': 'Sign, including billboards',
            '44': 'Chips, including wood chips',
            '45': 'Toy or game',
            '46': 'Awning, canopy',
            '47': 'Tarpaulin or tent',
            '5': 'Storage Supplies',
            '50': 'Storage supplies, other',
            '51': 'Box, carton, bag, basket, barrel',
            '52': 'Material being used to make a product',
            '53': 'Pallet, skid (empty)',
            '54': 'Cord, rope, twine',
            '55': 'Packing, wrapping material',
            '56': 'Baled goods or material',
            '57': 'Bulk storage',
            '58': 'Palletized material',
            '59': 'Rolled, wound material (paper, fabric)',
            '6': 'Liquids, Piping, Filters',
            '60': 'Liquids, piping, filters, other',
            '61': 'Atomized liquid, vaporized liquid, aerosol',
            '62': 'Flammable liquid/gas - in/from engine or burner',
            '63': 'Flammable liquid/gas - in/from final container',
            '64': 'Flammable liquid/gas in container or pipe',
            '65': 'Flammable liquid/gas - uncontained',
            '66': 'Pipe, duct, conduit or hose',
            '67': 'Pipe, duct, conduit, hose covering',
            '68': 'Filter, including evaporative cooler pads',
            '7': 'Organic Materials',
            '70': 'Organic materials, other',
            '71': 'Agricultural crop',
            '72': 'Light vegetation - not crop, including grass',
            '73': 'Heavy vegetation - not crop, including trees',
            '74': 'Animal living or dead',
            '75': 'Human living or dead',
            '76': 'Cooking materials, including edible materials',
            '77': 'Feathers or fur',
            '8': 'General Materials',
            '80': 'General Materials Other',
            '81': 'Electrical wire, cable insulation',
            '82': 'Transformer, including transformer fluids',
            '83': 'Conveyor belt, drive belt, V-belt',
            '84': 'Tire',
            '85': 'Railroad ties',
            '86': 'Fence, pole',
            '87': 'Fertilizer',
            '88': 'Pyrotechnics, explosives',
            '9': 'General Materials Continued',
            '90': 'General Materials Continued - Other',
            '91': 'Book',
            '92': 'Magazine, newspaper, writing paper',
            '93': 'Adhesive',
            '94': 'Dust, fiber, lint, including sawdust',
            '95': 'Film, residue, including paint & resin',
            '96': 'Rubbish, trash, or waste',
            '97': 'Oily rags',
            '99': 'Multiple items first ignited',
            'UU': 'Undetermined'
        }

        # 添加 MAT_SPRD 映射（与 TYPE_MAT_MAP 相同）
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
            '36': 'Combustible metal (e.g., magnesium)',
            '37': 'Solid chemical (e.g., explosives)',
            '38': 'Radioactive material',
            '4': 'Plastics',
            '41': 'Plastic',
            '5': 'Natural Product',
            '50': 'Natural product, other',
            '51': 'Rubber (excluding synthetic rubbers)',
            '52': 'Cork',
            '53': 'Leather',
            '54': 'Hay, straw',
            '55': 'Grain, natural fiber (preprocess)',
            '56': 'Coal, coke, briquettes, peat',
            '57': 'Food, starch (excluding fat and grease)',
            '58': 'Tobacco',
            '6': 'Wood or Paper - Processed',
            '60': 'Wood or paper, processed, other',
            '61': 'Wood chips, sawdust, shavings',
            '62': 'Round timber (posts, poles)',
            '63': 'Sawn wood (finished lumber)',
            '64': 'Plywood',
            '65': 'Fiberboard, particleboard, hardboard',
            '66': 'Wood pulp',
            '67': 'Paper (cellulose, waxed paper)',
            '68': 'Cardboard',
            '7': 'Fabric, Textiles, Fur',
            '70': 'Fabric, textile, fur, other',
            '71': 'Fabric, fiber (cotton, blends, rayon, wool)',
            '74': 'Fur, silk, other fabric',
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
            'UU': 'Undetermined'
        }
        # ========================
 
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
        if not self._is_missing(bi.get('square_footage')):
            lines.append(f"Area: The total projected area is <{bi.get('square_footage')}> square feet.")
        
        # 建筑构造信息 - 分别处理
        if not self._is_missing(bi.get('build_year')) and bi.get('build_year') != 'Unknown':
            lines.append(f"Construction Year: Constructed around the year <{bi.get('build_year')}>.")
        # if not self._is_missing(bi.get('build_material')) and bi.get('build_material') != 'Unknown':
        #     lines.append(f"Construction Material: The building is primarily made of <{bi.get('build_material')}>.")
        if not self._is_missing(bi.get('structure_type')) and bi.get('structure_type') != 'Unknown':
            lines.append(f"Structure Type: The building structure is <{bi.get('structure_type')}>.")

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
        lines.append("For any single-story building, the classification 'Fire confined in the floor' is equivalent to 'Fire confined in the building'.")
        
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
        if not self._is_missing(fi.get('suppression_factor_primary_desc')):
            lines.append(f"Suppression Factor: The main factor that hindered suppression efforts was <{fi.get('suppression_factor_primary_desc')}>.")
        lines.append("")
        
        # Community Context
        lines.append("% Community Context:")
        if not self._is_missing(cc.get('median_income_level')):
            lines.append(f"This building is situated in a community where the median income level is <{cc.get('median_income_level')}>.")
        if not self._is_missing(cc.get('median_rent_level')):
            lines.append(f"The median monthly rent level is <{cc.get('median_rent_level')}>.")
        if not self._is_missing(cc.get('housing_occupancy_level')):
            lines.append(f"Housing occupancy level is <{cc.get('housing_occupancy_level')}>.")
        if not self._is_missing(cc.get('bachelor_degree_level')):
            lines.append(f"Population with bachelor's degree or higher level is <{cc.get('bachelor_degree_level')}>. ")
        if not self._is_missing(cc.get('elderly_population_level')):
            lines.append(f"Elderly population (62+) level is <{cc.get('elderly_population_level')}>. ")
        if not self._is_missing(cc.get('black_population_level')):
            lines.append(f"Black or African American population level is <{cc.get('black_population_level')}>.")
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
        """将DataFrame转换为FireIncidentData对象列表（适配ny_data_balanced.csv列名）"""
        incident_data_list = []
        
        for _, row in df.iterrows():
            # 构建建筑信息
            building_info = {
                'state': row.get('STATE', 'Unknown'),
                'zip_code': str(row.get('ZIP5', 'Unknown')),
                'occupant_type': row.get('PROP_USE_new', row.get('PROP_USE', 'Unknown')),
                'occupant_type_desc': self.map_code(self.PROP_USE_MAP, row.get('PROP_USE_new', row.get('PROP_USE', 'Unknown')), 'Unknown'),
                'stories_above': row.get('BLDG_ABOVE', 'Unknown'),
                'stories_below': row.get('BLDG_BELOW', 'Unknown'),
                'num_units': row.get('NUM_UNIT', 'Unknown'),
                'square_footage': row.get('TOT_SQ_FT', 'Unknown'),
                'build_year': 'Unknown',  # 纽约数据中没有具体建筑年份信息
                # 'build_material': 'Enclosed Building' if str(row.get('STRUC_TYPE', '')).strip() == '1' else 'Unknown',
                'structure_type': self.map_code(self.STRUC_TYPE_MAP, row.get('STRUC_TYPE'), 'Unknown'),
                # 'on_site_material_major': row.get('ITEM_SPRD_new', 'Unknown'),
                # 'material_storage_use': row.get('MAT_SPRD_new', 'Unknown'),
                'on_site_material_major': self.map_code(self.ITEM_SPRD_MAP, row.get('ITEM_SPRD_new', row.get('ITEM_SPRD', 'Unknown')), 'Unknown'),
                'material_storage_use': self.map_code(self.MAT_SPRD_MAP, row.get('MAT_SPRD_new', row.get('MAT_SPRD', 'Unknown')), 'Unknown'),
            }

            
            # 构建天气信息
            weather_info = {
                'temperature': row.get('temp', 'Unknown'),
                'humidity': row.get('rhum', 'Unknown')
            }
            
            # 构建火灾信息
            fire_info = {
                'origin': row.get('AREA_ORIG_new', row.get('AREA_ORIG', 'Unknown')),
                'origin_desc': self.map_code(self.FIRE_ORIG_MAP, row.get('FIRE_ORIG'), self.map_code(self.AREA_ORIG_MAP, row.get('AREA_ORIG'), row.get('AREA_ORIG_new', 'Unknown'))),
                # 'time': f"{int(row.get('accident_hour', 0)):02d}h" if pd.notna(row.get('accident_hour')) else 'Unknown',
                # 'date': f"{int(row.get('accident_month', 1))}-{int(row.get('accident_year', 2012))}" if all(pd.notna(row.get(col)) for col in ['accident_month', 'accident_year']) else 'Unknown',
                'time': self._get_time_period(row.get('accident_hour')) if pd.notna(row.get('accident_hour')) else 'Unknown',
                'date': self._get_season(row.get('accident_month')) if pd.notna(row.get('accident_month')) else 'Unknown',
                # 'heat_source': row.get('HEAT_SOURCE_new', row.get('HEAT_SOURC', 'Unknown')),
                # 'heat_source_desc': self.map_code(self.HEAT_SOURC_MAP, row.get('HEAT_SOURC'), row.get('HEAT_SOURCE_new', 'Unknown')),
                # 'ignited_material': row.get('FIRST_IGN_new', row.get('FIRST_IGN', 'Unknown')),
                # 'ignited_material_desc': self.map_code(self.FIRST_IGN_MAP, row.get('FIRST_IGN'), row.get('FIRST_IGN_new', 'Unknown')),
                'heat_source': row.get('HEAT_SOURCE_new', row.get('HEAT_SOURC', 'Unknown')),
                'heat_source_desc': self.map_code(self.HEAT_SOURC_MAP, row.get('HEAT_SOURCE_new', row.get('HEAT_SOURC', 'Unknown')), 'Unknown'),
                'ignited_material': row.get('FIRST_IGN_new', row.get('FIRST_IGN', 'Unknown')),
                'ignited_material_desc': self.map_code(self.FIRST_IGN_MAP, row.get('FIRST_IGN_new', row.get('FIRST_IGN', 'Unknown')), 'Unknown'),
                # 'type_material_desc': self.map_code(self.TYPE_MAT_MAP, row.get('TYPE_MAT'), row.get('TYPE_MAT_new', 'Unknown')),
                'type_material_desc': self.map_code(self.TYPE_MAT_MAP, row.get('TYPE_MAT_new', row.get('TYPE_MAT', 'Unknown')), 'Unknown'),
                'floor': row.get('FIRE_ORIG', 'Unknown'),
                'fire_origin_floor': row.get('FIRE_ORIG', 'Unknown'),
                'detector_status': 'Present' if str(row.get('DETECTOR', '')).strip() in ['1', 'Y', 'y', 'Present', 'present'] else 'None Present',
                'detector_status_desc': 'Present' if str(row.get('DETECTOR', '')).strip() in ['1', 'Y', 'y', 'Present', 'present'] else 'None Present',
                'ase_status': 'Present' if str(row.get('AES_PRES', '')).strip() in ['1', 'Y', 'y', 'Present', 'present'] else 'None Present',
                'ase_status_desc': 'Present' if str(row.get('AES_PRES', '')).strip() in ['1', 'Y', 'y', 'Present', 'present'] else 'None Present',
                # 'item_contributing_spread_desc': row.get('ITEM_SPRD_new', row.get('ITEM_SPRD', 'Unknown')),
                # 'material_contributing_spread_desc': row.get('MAT_SPRD_new', row.get('MAT_SPRD', 'Unknown')),
                'item_contributing_spread_desc': self.map_code(self.ITEM_SPRD_MAP, row.get('ITEM_SPRD_new', row.get('ITEM_SPRD', 'Unknown')), 'Unknown'),
                'material_contributing_spread_desc': self.map_code(self.MAT_SPRD_MAP, row.get('MAT_SPRD_new', row.get('MAT_SPRD', 'Unknown')), 'Unknown'),
                'response_time': row.get('response_time', 'Unknown'),
                'human_factor_primary_desc': self.map_code(self.HUM1_MAP, row.get('hum_1'), 'Unknown'),
                'factor_ignition_primary_desc': self.map_code(self.FACT_IGN1_MAP, row.get('FACT_IGN1_new', row.get('FACT_IGN_1', 'Unknown')), 'Unknown'),
                'firefighter_action_primary_desc': self.map_code(self.ACT_TAK1_MAP, row.get('ACT_TAK1_new', row.get('ACT_TAK1', 'Unknown')), 'Unknown'),
                'suppression_factor_primary_desc': self.map_code(self.SUP_FAC1_MAP, row.get('SUP_FAC_1_new', row.get('SUP_FAC_1', 'Unknown')), 'Unknown'),
                'cause_ignition_desc': self.map_code(self.CAUSE_IGN_MAP, row.get('CAUSE_IGN'), 'Unknown'),
                'equipment_involved_desc': 'Unknown'  # 该字段在数据中可能不存在
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
    csv_path = "./data/processed_data/19_22_wo_ny_wash.csv"
    generator = FirePromptGenerator()
    
    # try:
    #     prompt = generator.generate_prompt_from_csv(csv_path)
    #     print("生成的示例prompt:")
    #     print(prompt)
    # except FileNotFoundError:
    #     print(f"未找到文件: {csv_path}")
    # except Exception as e:
    #     print(f"生成示例prompt失败: {e}")

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

