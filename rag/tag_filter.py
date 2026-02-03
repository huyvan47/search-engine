import re
import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Callable

# ===========================
# 1) PATH & LOAD KNOWLEDGE
# ===========================

BASE_DIR = Path(__file__).resolve().parent.parent
KB_PATH = BASE_DIR / "chemical_tags_from_kb.json"

with open(KB_PATH, "r", encoding="utf-8") as f:
    CHEMICAL_KB = json.load(f)

# ===========================
# 2) NORMALIZATION UTILITIES
# ===========================

_space_re = re.compile(r"\s+")
# Giữ + - / để không phá các tên chemical kiểu "pirimiphos-methyl", "s-metolachlor"
_non_alnum_keep_ops_re = re.compile(r"[^a-z0-9\s\+\-\/]+")
# Dùng cho entity match (crop/pest/disease/weed) để tránh mismatch "khoai-mi" vs "khoai mi"
_non_alnum_entity_re = re.compile(r"[^a-z0-9\s]+")

def _strip_accents_lower(text: str) -> str:
    text = text.lower().strip().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text


def normalize(text: str) -> str:
    """
    Normalize cho query/chemical: giữ + - / để không phá token chemical.
    """
    if not text:
        return ""

    text = (
        text.replace("\u00A0", " ")
        .replace("\u200b", " ")
        .replace("\t", " ")
        .replace("\n", " ")
    )
    text = _strip_accents_lower(text)

    # chỉ xóa các ký tự không cần thiết, GIỮ + - /
    text = _non_alnum_keep_ops_re.sub(" ", text)
    text = _space_re.sub(" ", text).strip()
    return text


def normalize_entity(text: str) -> str:
    """
    Normalize cho entity (crop/pest/disease/weed):
    - bỏ dấu
    - bỏ ký tự đặc biệt
    - coi '-' như khoảng trắng để match "khoai-mi" == "khoai mi"
    """
    if not text:
        return ""

    text = (
        text.replace("\u00A0", " ")
        .replace("\u200b", " ")
        .replace("\t", " ")
        .replace("\n", " ")
    )
    text = _strip_accents_lower(text)

    # loại bỏ toàn bộ dấu câu và ký tự đặc biệt
    text = _non_alnum_entity_re.sub(" ", text)
    text = text.replace("-", " ")  # an toàn nếu còn sót
    text = _space_re.sub(" ", text).strip()
    return text


def normalize_set(values: Set[str], normalizer: Callable[[str], str]) -> Set[str]:
    return {normalizer(x) for x in values if x}


# Chuẩn hóa sẵn dữ liệu trong KB theo entity-normalize để join ổn định
for _, v in CHEMICAL_KB.items():
    v["crops"] = [normalize_entity(x) for x in v.get("crops", [])]
    v["diseases"] = [normalize_entity(x) for x in v.get("diseases", [])]
    v["pests"] = [normalize_entity(x) for x in v.get("pests", [])]
    v["weeds"] = [normalize_entity(x) for x in v.get("weeds", [])]
    # formulation nếu có (nhiều KB không có)
    v["formulation"] = [normalize_entity(x) for x in v.get("formulation", [])]

# ===========================
# 3) ALIASES (BẠN TỰ COPY ĐẦY ĐỦ SAU)
# ===========================

CHEMICAL_ALIASES = {
    "24-epi-brassinolide": ["brassinolid 24-epi", "hooc mon brassinolide 24-epi", "brassinolide"],
    "24-epibrassinolide": ["brassinolid 24-epi", "hooc mon brassinolide 24-epi", "24-epibrassinolide"],
    "abamectin": ["abamectin", "tru sau abamectin"],
    "chlormequat": ["chat dieu hoa sinh truong chlormequat", "chlormequat"],
    "diuron": ["diuron", "diet co diuron"],
    "fomesafen": ["fomesafen", "diet co fomesafen"],
    "fosthiazate": ["fosthiazate", "diet sau fosthiazate", "gr", "rai goc", "rai goc"],
    "paclobutrazol": ["chat dieu hoa sinh truong paclobutrazol", "dieu hoa sinh truong paclobutrazol", "paclobutrazol"],
    "profenofos": ["profenofos", "tru sau profenofos"],
    "spirotetramat": ["spirotetramat", "tru sau spirotetramat"],
    "sulfur": ["chat diet nam sulfur", "luu huynh", "sulfur"],
    "abinsec-oxatin-1-8ec": ["abinsec oxatin 1 8ec", "abinsec oxatin 1 8ec"],
    "abound": ["abound", "diet nam abound"],
    "acetamiprid": ["acetamiprid", "tru sau acetamiprid"],
    "acetochlor": ["acetoclor", "acetochlor"],
    "alpha-cypermethrin": ["alpha cypermethrin", "alpha xi permethrin", "permethrin", "cypermethrin", "alpha-cypermethrin"],
    "ametryn": ["ametrin", "ametryn"],
    "atrazine": ["atrazin", "atrazine"],
    "avermectin": ["avermectin"],
    "avermectin-b1a": ["avermectin b1a", "avermectin"],
    "avermectin-b1b": ["avermectin b1b", "avermectin"],
    "azol-450sc": ["azol 450sc"],
    "azoxystrobin": ["azoxistrobin", "azoxystrobin"],
    "azoxystrobincrop:rice": ["azoxystrobin lua"],
    "azoxytrobin": ["azoxistrobin", "azoxytrobin"],
    "bacillus-subtilis": ["bacillus subtilis", "bacillus-subtilis", "bacillus", "subtilis"],
    "badge-x2": ["badge x2"],
    "bentazone": ["bentazon", "bentazone"],
    "beta-cypermethrin": ["beta cypermethrin", "beta-cypermethrin", "cypermethrin", "permethrin"],
    "bifenazate": ["bifenazat", "bifenazate"],
    "bifenthrin": ["bi fen trin", "bifenthrin"],
    "bismerthiazol": ["bismerthiazol"],
    "bismerthiazole": ["bismerthiazol", "bismerthiazole"],
    "boscalid": ["boscalid"],
    "brodifacoum": ["brodifacoum"],
    "buprofezin": ["bu profezin", "buprofezin"],
    "c10-alcohol-ethoxylate": ["c10 alcohol ethoxylate", "alcohol", "ethoxylate", "c10"],
    "cabona": ["diet sau cabona", "tru sau cabona"],
    "calcium-hydroxide": ["bot canxi", "canxi hydroxit", "tri benh canxi"],
    "calcium-nitrate": ["canxi nitrat", "phan bon canxi nitrat"],
    "cartap-hydrochloride": ["cartap hydroclorua", "diet sau cartap", "cartap", "hydrochloride", "cartap-hydrochloride"],
    "chloramphenicol": ["cloramphenicol", "khang sinh cloramphenicol"],
    "chlorfenapyr": ["clorfenapyr", "diet sau clorfenapyr", "chlorfenapyr"],
    "chlorothalonil": ["clorotalonil", "tri nam clorotalonil", "chlorothalonil"],
    "chlorothalonil-50sc": ["clorotalonil 50sc", "tri nam clorotalonil 50sc", "chlorothalonil"],
    "chlorothalonil-75wp": ["clorotalonil 75wp", "tri nam clorotalonil 75wp", "chlorothalonil"],
    "chlorpyrifos-ethyl": ["clorpyrifos etyl", "diet sau clorpyrifos etyl", "chlorpyrifos"],
    "chlorpyrifos-methyl": ["clorpyrifos metyl", "diet sau clorpyrifos metyl", "chlorpyrifos", "chlorpyrifos-methyl"],
    "contact-fungicide": ["diet nam tiep xuc", "tri nam tiep xuc"],
    "copper-based-fungicide": ["diet nam goc dong", "tri nam goc dong"],
    "copper-based-pesticide": ["diet sau goc dong", "tru sau goc dong"],
    "copper-chelate": ["dong chelate", "dong hoa hop chelate"],
    "copper-fungicide": ["diet nam dong", "tri nam dong"],
    "copper-hydroxide": ["dong bot hydroxit", "dong hydroxit"],
    "copper-ii-hydroxide": ["dong hydroxit", "dong ii hydroxit"],
    "copper-ion": ["dong ion", "ion dong"],
    "copper-oxychloride": ["dong oxyclorid", "dong oxyclorua"],
    "copper-sulfate": ["dong sulfat", "dong sunfat"],
    "cuprous-oxide": ["dong i oxit", "dong oxit"],
    "cyazofamid": ["cyazofamid", "diet nam cyazofamid"],
    "cyclopiazonic-acid": ["axit cyclopiazonic", "chat doc cyclopiazonic"],
    "cymoxanil": ["cymoxanil", "diet nam cymoxanil"],
    "cypermethrin": ["cypermethrin", "diet sau cypermethrin", "tru sau cypermethrin"],
    "cyromazine": ["cyromazine", "diet sau cyromazine"],
    "daconil": ["daconil", "tri nam daconil"],
    "dcpa": ["dcpa", "diet co dcpa"],
    "dieu-hoa-sinh-truong": ["dieu hoa sinh truong", "dieu hoa", "sinh truong"],
    "deltamethrin": ["deltamethrin", "diet sau deltamethrin"],
    "diafenthiuron": ["diafenthiuron", "diet sau diafenthiuron"],
    "dichloran": ["dichloran", "diet nam dichloran"],
    "difenoconazole": ["difenoconazole", "tri nam difenoconazole"],
    "dimethoate": ["dimethoate", "diet sau dimethoate"],
    "dimethomorph": ["dimethomorph", "tri nam dimethomorph"],
    "dinotefuran": ["diet sau dinotefuran", "tru sau dinotefuran", "dinotefuran"],
    "diquat-dibromide": ["diet co diquat", "diet co diquat dibromide", "diquat-dibromide"],
    "disease-control-chemical": ["diet benh", "phong tru benh"],
    "disease-fungicide-group-a": ["diet nam nhom a"],
    "dithianon": ["diet nam dithianon", "dithianon"],
    "emamectin-benzoate": ["diet sau emamectin benzoate", "tru sau emamectin benzoate", "emamectin", "benzoate", "emamectin-benzoate"],
    "enable": ["diet sau enable"],
    "ethanol-70": ["con 70 do", "cuu am 70 do", "ethanol 70 do"],
    "ethephon": ["kich thich ethephon", "kich thich ra hoa ethephon", "ethephon"],
    "ethyl-alcohol": ["con etylic", "cuu am etylic"],
    "ethyl-alcohol-70": ["con etylic 70 do", "cuu am etylic 70 do"],
    "fenbuconazole": ["diet nam fenbuconazole"],
    "fenclorim": ["diet co fenclorim"],
    "fenobucarb": ["diet sau fenobucarb", "fenobucarb"],
    "fertilizer": ["phan bon", "phan hoa hoc"],
    "flonicamid": ["diet sau flonicamid", "flonicamid"],
    "fluazinam": ["diet nam fluazinam", "fluazinam"],
    "fludioxonil": ["diet nam fludioxonil"],
    "fluopicolide": ["diet nam fluopicolide", "fluopicolide"],
    "flusilazole": ["diet nam flusilazole", "flusilazole"],
    "forgon-40ec": ["diet sau forgon 40ec"],
    "forsan-60ec": ["diet sau forsan 60ec"],
    "fosphite": ["phan fosfit", "phan fosfit truyen dinh duong"],
    "fullkill-50ec": ["diet sau fullkill 50ec"],
    "gibberellic-acid": ["acid gibberellic", "chat kich thich tang truong gibberellic", "gibberellic", "gibberellic-acid"],
    "glufosinate-amonium": ["diet co glufosinate amonium", "glu", "glufosinate", "amonium", "glufosinate-amonium"],
    "glufosinate-p": ["diet co glufosinate p", "glufosinate-p"],
    "haloxyfop-p-methyl": ["diet co haloxyfop p methyl", "haloxyfop", "haloxyfop-p-methyl"],
    "headline": ["diet sau headline"],
    "heritage": ["tru sau heritage"],
    "hexaconazole": ["tru benh hexaconazole", "tru nam hexaconazole", "hexaconazole"],
    "hexythiazox": ["tru tru sau hexythiazox", "hexythiazox"],
    "hoagland-solution": ["dung dich hoagland"],
    "hoaglands-solution": ["dung dich hoagland"],
    "hymexazol": ["tru nam hymexazol", "hymexazol"],
    "kasugamycin": ["kasugamycin", "khang sinh kasugamycin", "kasugamycin"],
    "ic-top": ["tru sau ic-top"],
    "imazalil": ["tru nam imazalil"],
    "imidacloprid": ["tru sau imidacloprid", "tru sau imidakloprid", "imidacloprid"],
    "iprodione": ["tru benh iprodione"],
    "isopropyl-alcohol": ["con isopropanol", "con isopropyl"],
    "isoprothiolane": ["tru benh isoprothiolane", "isoprothiolane"],
    "jingangmycin": ["khang sinh jingangmycin", "jingangmycin"],
    "kresoxim-methyl": ["tru benh kresoxim methyl", "kresoxim", "kresoxim-methyl"],
    "lactofen": ["diet co lactofen", "lactofen"],
    "lambda-cyhalothrin": ["tru sau lambda cyhalothrin", "lambda", "cyhalothrin", "lambda-cyhalothrin"],
    "lufenuron": ["tru sau lufenuron", "lufenuron"],
    "mancozeb": ["tru benh mancozeb", "mancozeb", "manco"],
    "mefenoxam": ["tru benh mefenoxam"],
    "mesotrione": ["diet co mesotrione", "mesotrione"],
    "metaflumizone": ["tru sau metaflumizone", "metaflumizone"],
    "metaldehyde": ["tru sau metaflumizone", "metaflumizone", "metaldehyde"],
    "metalaxyl": ["tru benh metalaxyl", "metalaxyl"],
    "oxine-copper": ["oxine dong", "diet sau oxine dong"],
    "penconazole": ["penconazole", "diet nam penconazole"],
    "peptone": ["chat dinh duong peptone", "chat peptone"],
    "permethrin": ["permethrin", "diet sau permethrin"],
    "phenthoate": ["phenthoate", "diet sau phenthoate"],
    "phoxim": ["phoxim", "diet sau phoxim"],
    "pirimiphos-methyl": ["pirimiphos methyl", "diet sau pirimiphos methyl", "pirimiphos", "pirimiphos-methyl", "gb", "ba moi", "oc rai", "rai goc"],
    "pretilachlor": ["tru sau pretilachlor", "pretilachlor"],
    "pristine": ["tru nam pristine"],
    "probiconazole": ["tru nam probiconazole", "probiconazole"],
    "prochloraz": ["tru nam prochloraz"],
    "prochloraz-manganese-chloride-complex": ["phuc hop prochloraz mangan clorua"],
    "prochloraz-manganese-complex": ["phuc hop prochloraz mangan", "prochloraz"],
    "prochloraz-manganesse-complex": ["phuc hop prochloraz mangan", "prochloraz", "manganesse"],
    "propamocarb": ["tru nam propamocarb", "propamocarb"],
    "propamocarb-hcl": ["propamocarb hydrochloride", "propamocarb", "hydrochloride", "propamocarb-hcl"],
    "propiconazole": ["tru nam propiconazole", "propiconazole"],
    "propoxur": ["tru sau propoxur", "propoxur", "gb", "ba moi", "oc rai", "rai goc"],
    "proteose-peptone": ["proteose pepton"],
    "pymetrozine": ["tru sau pymetrozine", "pymetrozine"],
    "pyraclostrobin": ["tru nam pyraclostrobin", "pyraclostrobin"],
    "pyrethroid": ["phan bo pyrethroid", "tru sau pyrethroid", "pyrethroid", "pyrethroid"],
    "pyridaben": ["tru sau pyridaben", "pyridaben"],
    "pyriproxyfen": ["tru sau pyriproxyfen", "pyriproxyfen"],
    "quizalofop-p-ethyl": ["quizalofop p ethyl", "quizalofop", "quizalofop-p-ethyl"],
    "r333": ["tru sau r333"],
    "s-metolachlor": ["tru sau s metolachlor", "metolachlor", "s-metolachlor"],
    "spirodiclofen": ["tru sau spirodiclofen", "spirodiclofen"],
    "tebuconazole": ["tebuconazole", "tri nam tebuconazole", "tebuconazole"],
    "tembotrione": ["diet co tembotrione", "tembotrione"],
    "terbuthylazine": ["diet co terbuthylazine", "terbuthylazine"],
    "terrachlor": ["diet co terrachlor"],
    "than-duoc-sach-benh": ["than duoc sach benh", "sach benh"],
    "thiabendazole": ["thiabendazole", "tri nam thiabendazole"],
    "thiacloprid": ["diet sau thiacloprid", "thiacloprid"],
    "thiamethoxam": ["diet sau thiamethoxam", "thiamethoxam"],
    "thiosultap-sodium": ["diet sau thiosultap sodium", "thiosultap", "thiosultap-sodium", "gr", "rai goc", "rai goc"],
    "thiram": ["thiram", "tri nam thiram", "thiram"],
    "tolfenpyrad": ["diet sau tolfenpyrad", "tolfenpyrad", "tolfenpyrad"],
    "topramezone": ["diet co topramezone", "topramezone", "topramezone"],
    "toxic-chemical": ["chat doc", "chat doc hai"],
    "triadimefon": ["tri nam triadimefon", "triadimefon", "triadimefon"],
    "tricyclazole": ["tri nam tricyclazole", "tricyclazole", "tricyclazole"],
    "tricylazole": ["tri nam tricylazole", "tricylazole", "tricylazole"],
    "trifloxystrobin": ["tri nam trifloxystrobin", "trifloxystrobin"],
    "trinong-50wp": ["tri nam trinong 50wp", "trinong 50wp"],
    "trioxystrobin": ["tri nam trioxystrobin", "trioxystrobin", "trioxystrobin"],
    "ultra-flourish": ["phan bon ultra flourish", "ultra flourish"],
    "wa-0-05": ["wa 0 05"],
    "zearalenone": ["chat doc zearalenon", "chat doc zearalenone"],
    "zhongshengmycin": ["khang sinh zhongshengmycin", "zhongshengmycin", "zhongshengmycin"],
    "xu li hat giong": ["xu li hat giong", "hat giong"],
}

CROP_ALIASES = {
    "apple": ["cay tao", "tao"],
    "avocado": ["qua bo", "cay bo", "trai bo"],
    "bau": ["bau"],
    "banana": ["cay chuoi", "chuoi"],
    "bap": ["bap", "cay bap", "corn", "ngo"],
    "bap-cai": ["bap cai", "cay bap cai"],
    "barley": ["cay lua mi", "lua mi"],
    "bean": ["cay dau", "dau"],
    "kho-qua": ["kho qua", "qua kho qua", "muop dang"],
    "tieu": ["tieu", "tieu den"],
    "bi-dao": ["bi dao", "bi huong"],
    "buoi": ["buoi"],
    "bong-vai": ["bong vai"],
    "ca-chua": ["ca chua", "ca chua bi"],
    "ca-phe": ["ca phe"],
    "rau-cai-thia": ["rau cai thia"],
    "cai-brussel": ["cai brussel"],
    "cai": ["cai"],
    "cai-cu": ["cai cu"],
    "cai-thao": ["cai thao"],
    "can-nuoc": ["can nuoc"],
    "can-tay": ["can tay"],
    "ca-phao": ["ca phao"],
    "cac-loai-dua": ["cac loai dua"],
    "cac-loai-hoa": ["cac loai hoa"],
    "cay-mo": ["cay mo"],
    "cay-rung": ["cay rung"],
    "chanh-day": ["chanh day"],
    "chanh-quyt": ["chanh quyt"],
    "che": ["che"],
    "tra": ["tra"],
    "cu-cai": ["cu cai"],
    "dau-cove": ["dau cove"],
    "dau-den": ["dau den"],
    "dau-tam": ["dau tam"],
    "dat-khong-trong-trot": ["dat khong trong trot"],
    "dat-trong-cay-cong-nghiep": ["dat trong cay cong nghiep", "cay cong nghiep"],
    "dua-le": ["dua le"],
    "le": ["le"],
    "dua-leo": ["dua leo", "dua chuot"],
    "dua-luoi": ["dua luoi"],
    "mac-ca": ["mac ca"],
    "mang-tay": ["mang tay"],
    "mong-toi": ["mong toi"],
    "tan-o": ["tan o"],
    "ca-tim": ["ca tim"],
    "cacao": ["ca cao"],
    "cam": ["cam"],
    "cay-an-qua": ["cam", "quyt", "buoi", "xoai", "cafe", "ca phe"],
    "cam-quyt": ["cam quyt"],
    "cao-su": ["cao su"],
    "carrot": ["ca rot"],
    "san": ["san"],
    "xa-lach": ["xa lach"],
    "thom": ["thom"],
    "tia-to": ["tia to"],
    "cu-toi": ["cu toi"],
    "cay-an-trai": ["cay an trai"],
    "cay-cam": ["cay cam"],
    "cay-con": ["cay con"],
    "cay-trong": ["cay trong"],
    "chanh": ["chanh"],
    "che": ["che"],
    "ca-cao": ["ca cao"],
    "bong-vai": ["bo vai"],
    "cu-dau": ["cu dau", "cu dau nong nghiep"],
    "dau-nanh": ["dau nanh", "hat dau nanh"],
    "dau-phong": ["dau phong", "hat dau phong"],
    "dau-tay": ["dau tay", "hat dau tay"],
    "dau-tuong": ["dau tuong", "hat dau tuong"],
    "dau-xanh": ["dau xanh", "hat dau xanh"],
    "dieu": ["cay dieu", "hat dieu"],
    "dragon-fruit": ["cay thanh long", "thanh long"],
    "du-du": ["cay du du", "du du"],
    "dua": ["cay dua", "dua"],
    # "durian": ["cay sau rieng", "sau rieng"],
    "ca-tim": ["ca tim", "cay ca tim"],
    "hoa": ["cay hoa"],
    "sup-lo": ["sup lo"],
    "gung": ["gung"],
    "hanh": ["hanh"],
    "hanh-la": ["hanh la"],
    "hoa-cuc": ["hoa cuc"],
    "hoa-dao": ["hoa dao"],
    "hoa-hong": ["hoa hong"],
    "hanh-hoa": ["hanh hoa"],
    "hanh-tay": ["hanh tay"],
    "oi": ["oi"],
    "hoa-hong-dai-multiflora": ["hoa hong dai"],
    "hoa-ly": ["hoa ly"],
    "hoa-mai": ["hoa mai"],
    "mai": ["mai"],
    "jackfruit": ["mit"],
    "khoai-lang": ["khoai lang"],
    "khoai-tay": ["khoai tay"],
    "khoai-mi": ["khoai mi", "khoai san"],
    "khom": ["khom"],
    "lac": ["dau phong", "lac"],
    "chanh-vang": ["chanh vang"],
    "lime": ["chanh tay", "lime"],
    "longan": ["nhan"],
    "nghe": ["nghe"],
    "lua": ["cay lua", "lua"],
    "mai": ["hoa mai"],
    "mang-cau": ["mang cau"],
    "me-vung": ["me vung"],
    "mia": ["mia"],
    "mit": ["mit"],
    "ngo": ["ngo"],
    "nho": ["nho"],
    "ot": ["ot"],
    "pak-choi": ["cai bap"],
    "pea": ["dau ha lan"],
    "peach": ["cay dao"],
    "peanut": ["dau phong", "hat dau phong"],
    "pear": ["cay le", "le"],
    "pepper": ["cay tieu", "tieu"],
    "dua-hau": ["dua hau", "dua tay"],
    "plum": ["cay man", "man"],
    "potato": ["khoai", "khoai tay"],
    "quyt": ["cay quyt", "quyt"],
    "ra-hoa": ["ra hoa", "thoi ky ra hoa"],
    "quat": ["quat"],
    "tac": ["tac"],
    "rau-cai": ["rau cai", "rau cai la"],
    "rau-muong": ["rau muong"],
    "rau-mau": ["rau mau", "rau mau an"],
    "rau-cu": ["rau cu"],
    "rau-ho-thap-tu": ["rau ho thap tu", "ho thap tu"],
    "rau-den": ["rau den"],
    "rau-ngo": ["rau ngo"],
    "rau-diep": ["rau diep"],
    "rau-trai": ["rau trai"],
    "thai-lai": ["thai lai"],
    "rau-xa-bong": ["rau xa bong"],
    "rau-chan-vit": ["rau chan vit", "chan vit"],
    "resistant-variety": ["giong chong benh", "giong khang benh"],
    "lua-sa": ["lua sa"],
    # "rose": ["cay hoa hong", "hoa hong"],
    "sam-ngoc-linh": ["sam ngoc linh"],
    "san": ["cay san"],
    "su-hao": ["su hao"],
    "sau-rieng": ["cay sau rieng", "sau rieng"],
    "bau-bi": ["bi ngo", "qua bi ngo"],
    "sweet-orange": ["cam ngot"],
    "thanh-long": ["qua thanh long", "thanh long"],
    "thuoc-la": ["thuoc la"],
    "vegetable": ["rau an"],
    "vegetable-crops": ["cay rau"],
    "vegetables": ["rau an"],
    "vuon-cay-an-trai": ["vuon cay an trai"],
    "watermelon": ["dua hau", "qua dua hau"],
    "long-vuc": ["long vuc"],
    "wheat": ["lua mi"],
    "xoai": ["qua xoai", "xoai"],
}

DISEASE_ALIASES = {
    "nhom-a": 
    ["nhom benh a", "nhom a", "nam nhom a", "nhom nam a", "benh nhom a", "than thu", "dom vong", "dom tim", "bi thoi", "chay la", "dom nau", "dom la", "heo ru", "chet cham", "chay day", "thoi re", "lua von", "lem lep hat", "phan trang", "moc xam", "nam long chuot", "ghe la", "ghe trai", "dom den", "thoi than", "thoi hach", "thoi re", "benh thoi canh", "chay canh", "thoi qua", "benh chet canh", "benh scab", "benh ghe", "san vo", "tiem lua", "vang be", "thoi trai", "kho dot", "chet canh", "nut than", "chay nhua", "benh dom nau", "kho", "benh thoi", "vet nut"
    ],
    "nhom-b": 
    ["nhom benh b", "nhom b", "nam nhom b", "nhom nam b", "benh nhom b", "lo co re", "heo cay con", "chay la", "kho van", "nam hong", "heo ru", "moc trang", "co re bi thoi nau", "thoi nau", "thoi nhun", "benh chet rap cay con", "thoi trai", "thoi than", "ri sat", "than hat lua", "benh ri sat dau tuong", "than thu", "dom la lon", "lem lep hat", "benh thoi"
    ],
    "nhom-o": 
    ["nhom benh o", "nhom o", "nam nhom o", "nhom nam o", "benh nhom o", "suong mai", "benh thoi re", "thoi ngon", "thoi mam", "chet nhanh", "thoi trai", "nut than", "xi mu", "vang la", "chet than", "chet canh", "thoi re", "chet cay con", "moc suong", "gia suong mai", "soc trang", "bach tang", "moc xuong", "ri trang", "nam trang", "phong trang", "benh thoi"],
    "ba-trau": ["dom ba trau", "dom nau", "ba trau", "vet nut"],
    "bac-la": ["bac la", ],
    "benh-nam-hoa-vang": ["hoa vang",],
    "chay-bia-la": ["chay bia la", ],
    "chet-day": ["chet day", ],
    "dom-la": ["dom den la", "den la", "dom la"],
    "dom-den-la": ["dom den la", "den la", "dom la"],
    "dom-mat-cua": ["mat cua"],
    "dom-trai": ["dom trai"],
    "dom-van": ["dom van"],
    "dom-vang": ["dom vang"],
    "dao-on": ["benh dao on", "dao on", "dao on co bong"],
    "heo-vang": ["heo vang"],
    "heo-xanh": ["heo xanh"],
    "lem-lep": ["lem lep"],
    "loet": ["loet"],
    "rung-la-mai": ["rung la mai"],
    "rung-la": ["rung la"],
    "ghe": ["ghe"],
    "seo": ["seo"],
    "ghe-khoai": ["ghe khoai"],
    "ghe-cam": ["ghe cam"],
    "ghe-seo": ["ghe seo"],
    "seo-qua": ["seo qua"],
    "seo-trai": ["seo trai"],
    "thoi-co-gie": ["thoi co gie"],
    "thoi-cu": ["thoi cu"],
    "thoi-dau-trai": ["thoi dau trai"],
    "rung-trai": ["rung trai"],
    "thoi-den": ["thoi den"],
    "thoi-goc": ["thoi goc"],
    "tuyen-trung": ["tuyen trung"],
    "thoi-than-xi-mu": ["thoi than xi mu"],
    "u-cuc-re": ["u cuc re"],
    "vang-la-chin-som": ["vang la chin som"],
    "vang-rung-la": ["vang rung la"],
    "xoan-la": ["xoan la"],

}

PEST_ALIASES = {
    "bo-ngau": ["bo ngau", "bo ngau tren cay"],
    "bo-nhay": ["bo nhay", "bo nhay tren cay"],
    "bo-phan": ["bo phan", "bo phan tren cay"],
    "bo-hung": ["bo hung", "bo phan tren cay"],
    "bo-rua": ["bo rua",],
    "bo-canh-to": ["bo canh to",],
    "bu-lach": ["bu lach",],
    "bo-tri-vang": ["bo tri vang",],
    "cay-xau-ho": ["cay xau ho"],
    "chac-lac": ["chac lac"],
    "buom-trang-hai-cham": ["buom trang hai cham",],
    "bo-phan-trang": ["bo phan trang", "bo phan trang tren cay"],
    "bo-tri": ["bo tri", "xu ly tri", "xu ly bo tri", "bo hut", "bo tri tren cay"],
    "bo-xit": ["bo xit", "bo xit tren cay", "bo xit muoi"],
    "bo-ha": ["bo ha", "sung dat", "sung"],
    "borer": ["sau duc than", "sau duc trong"],
    "borers": ["sau duc than", "sau duc trong"],
    "co-la-rong": ["co la rong", "co la rong trong ruong"],
    "co-duoi-chon": ["co duoi chon",],
    "co-dong-tien": ["co dong tien",],
    "co-hoi": ["co hoi",],
    "co-long-hoi": ["co long hoi",],
    "co-la-tre": ["co la tre",],
    "co-cu": ["co cu",],
    "co-lac": ["co lac",],
    "co-muc": ["co muc",],
    "co-gao": ["co gao",],
    "co-ray": ["co ray",],
    "co-tuc": ["co tuc",],
    "co-gau": ["co gau",],
    "co-chac": ["co chac",],
    "co-hoa-ban": ["co hoa ban",],
    "co-hoa-thao": ["co hoa thao",],
    "co-chao": ["co chao", "co chao trong ruong"],
    "co-chi": ["co chi", "co chi trong ruong"],
    "co-dai-la-hep": ["co dai la hep", "co dai la nho", "la hep"],
    "co-dai-la-rong": ["co dai la rong", "co dai la to", "la rong"],
    "co-duoi-phung": ["co duoi phung", "co duoi phung la"],
    "man-trau": ["man trau"],
    "co-man-trau": ["co man trau", "co man trau la", "man-trau"],
    "co-tranh": ["co tranh", "co tranh la"],
    "con-trung": ["con trung"],
    "compacted-soil": ["dat nen cat cung", "dat nen cung"],
    "doi-duc-la": ["benh doi duc la", "doi duc la"],
    "duc-than": ["duc than"],
    "duoi-phung": ["duoi phung"],
    "eggs": ["trung con trung", "trung sau"],
    "sieu-nhan": ["sieu nhan"],
    "glyphosate-resistant-weeds": ["cac loai co khang glyphosate", "co khang", "khang glyphosate"],
    "oc-buou-vang": ["oc buou", "oc buou vang", "oc vang"],
    "nam": ["nam", "nam cay trong"],
    "nhen": ["con nhen", "ruoi nhen", "nhen"],
    "oc-buu-vang": ["oc buu vang", "oc vang","oc"],
    "pests": ["sau benh", "sau hai"],
    "ph": ["do axit kiem", "do ph"],
    "phan-trang": ["benh phan trang", "phan trang"],
    "rat": ["chuot hai cay"],
    "ray": ["ray", "ray tren cay"],
    "ray-bong": ["ray bong", "ray bong tren cay"],
    "lung-trang": ["lung trang",],
    "ray-lung-trang": ["ray lung trang", "ray lung trang tren cay"],
    "ray-mem": ["ray mem", "ray mem tren cay"],
    "ray-nau": ["ray nau", "ray lung trang", "ray nau tren cay"],
    "ray-phan": ["ray phan", "ray phan tren cay"],
    "ray-xanh": ["ray xanh", "ray-chong-canh", "ray phan", "ray mem", "bo phan trang", "bo xit"],
    "rep": ["rep", "rep tren cay"],
    "rep-bong-xo": ["rep bong xo", "rep bong xo tren cay"],
    "rep-mem": ["rep mem", "rep mem tren cay"],
    "rep-muoi": ["rep muoi", "rep muoi tren cay"],
    "rep-sap": ["rep sap", "rep vay", "rep sap tren cay"],
    "rep-vay": ["rep vay", "rep sap", "rep vay tren cay"],
    "ruoi-vang": ["ruoi vang", "ruoi vang lua"],
    "rust-fungus": ["benh nam giang", "nam giang"],
    "sau-benh": ["sau benh", "sau gay benh"],
    "sau-bo": ["sau bo", "sau bo la"],
    "sau-chich-hut": ["sau chich hut", "sau hut mau"],
    "sau-cuon-la": ["sau cuon la", "sau cuon la lua", "cuon la"],
    "sau-duc-hoa": ["sau duc hoa"],
    "sau-duc-ngon": ["sau duc ngon"],
    "sau-keo": ["sau keo"],
    "sau-long": ["sau long"],
    "sau-dat": ["sau dat", "sau dat lua"],
    "sau-nang": ["sau nang"],
    "sau-phao": ["sau phao"],
    "sau-rom": ["sau rom"],
    "sen-nhot": ["sen nhot"],
    "sen-vo-mong": ["sen vo mong"],
    "sung-bo-ha": ["sung bo ha"],
    "sau-xanh-da-lang": ["sau xanh da lang", "da lang", "sau xanh"],
    "sau-duc-be": ["sau duc be", "sau duc than be"],
    "sau-duc-canh": ["sau duc canh", "sau duc canh lua"],
    "sau-duc-qua": ["sau duc qua", "sau duc qua lua"],
    "sau-duc-than": ["sau duc than", "sau duc than lua"],
    "sau-duc-trong": ["sau duc trong", "sau duc trong lua"],
    "sau-hai": ["sau hai", "sau hai lua"],
    "sau-hanh": ["sau hanh", "sau cho hanh", "sau hanh lua"],
    "sau-khoang": ["sau khoang", "sau khoang lua"],
    "sau-mieng-nhai": ["sau mieng nhai", "sau nhai la"],
    "sau-nan": ["sau nan", "sau nan lua"],
    "sau-rieng-fruit-borer": ["sau duc trai qua rieng"],
    "sau-sung-trang": ["sau sung trang", "sau sung trang lua"],
    "sau-to": ["sau to", "sau to tren bap cai", "sau to tren bap su"],
    "sau-tong-hop": ["cac loai sau tong hop", "sau tong hop"],
    "sau-ve-bua": ["ve bua"],
    "sau-xam": ["sau xam", "sau xam la"],
    "sau-xanh": ["sau xanh", "sau xanh la"],
    "slug": ["oc ban", "oc ban trong ruong"],
    "oc-sen": ["oc sen", "oc sen trong ruong"],
    "oc-nhot": ["oc nhot"],
    "sau-bay": ["sau bay"],
    "xi-mu": ["xi mu"],
    "nut-vo": ["nut vo"],
    "nut-than": ["nut than"],
    "sung-khoai": ["sung khoai", "sung khoai tay"],
    "suong-mai": ["benh suong mai", "suong mai"],
    "than-thu": ["sau than", "than thu"],
    "ve-sau": ["con ve sau", "ve sau"],
    "weeds": ["co dai", "mac-co", "rau sam", "co cuc", "cho de", "den gai", "co chan vit", "co long vuc", "co man trau"],
    "mac-co": ["mac co"],
    "mat-cua": ["mat cua"],
    "moi": ["diet moi", "tru moi", "moi mot"],
    "mot-duc-canh": ["mot duc canh", "mot"],
    "muoi-den": ["muoi den"],
    "nam-coc": ["nam coc"],
    "nam-hoa-vang": ["nam hoa vang"],
    "nam-moc-nau": ["nam moc nau"],
    "ray-chong-canh": ["ray chong canh"],
    "tam-bop": ["tam bop"],
}

PRODUCT_ALIASES = {
    "afenzole-top-325sc": ["afenzole", "diet sau afenzole top", "tru sau afenzole", "afenzoletop", "2 hoat chat", "hai hoat chat"],
    "amamectin-60": ["diet sau amamectin", "tru sau amamectin 60", "amamectin"],
    "anh-hung-sau": ["sau anh hung", "sau hai hai", "anh hung sau"],
    "ankamec-3.6ec": ["diet sau ankamec", "tru sau ankamec 3 6ec", "ankamec"],
    "asmilka-top-325sc": ["diet sau asmilka top", "tru sau asmilka top 325sc", "asmilka"],
    "atomin-15wp": ["diet sau atomin", "tru sau atomin 15wp", "atomin"],
    "benxana-240sc": ["diet sau benxana", "tru sau benxana 240sc", "benxana"],
    "bmc-sulfur-80wg": ["diet nam bmc sulfur", "tru nam bmc sulfur 80wg", "sulfur"],
    "bo-rung": ["bo rung", "con bo rung"],
    "bong-ra-sai-150": ["diet sau bong ra sai", "tru sau bong ra sai 150", "bong ra sai","bongrasai", "bongsai"],
    "dapharnec-3.6ec": ["diet sau dapharnec", "tru sau dapharnec 3 6ec", "dapharnec"],
    "downy-650wp": ["diet nam downy", "tru nam downy 650wp", "downy"],
    "dum-xpro-650wp": ["dum xpro", "dum pro", "diet nam dum xpro", "tru nam dum xpro 650wp", "xpro", "dumxpro"],
    "forsan-60ec": ["diet sau forsan", "tru sau forsan 60ec", "forsan", "forsan60ec"],
    "g9-thanh-sau": ["diet sau g9 thanh sau", "tru sau g9 thanh sau", "thanh sau", "g9"],
    "giao-su-benh-4.0": ["diet nam giao su benh", "tru nam giao su benh", "giao su benh"],
    "gone-super-350ec": ["gone super", "gon super", "diet sau gone super", "tru sau gone super 350ec", "gon super", "gone super", "3 hoat chat", "ba hoat chat"],
    "haihamec": ["diet sau haihamec", "tru sau haihamec", "haihamec"],
    "haruko-5sc": ["diet sau haruko", "tru sau haruko 5sc", "haruko"],
    "haseidn-gold": ["diet sau haseidn gold", "tru sau haseidn gold", "haseidn"],
    "horisan-75": ["diet nam horisan", "tru nam horisan 75", "horisan"],
    "kaijo-5.0wg": ["diet nam kaijo", "tru nam kaijo 5 0wg", "kaijo"],
    "kajio-1gr-alpha": ["diet sau kajio 1gr alpha", "diet sau kajio alpha", "kaijo"],
    "kajio-1gr-gold": ["diet sau kajio 1gr gold", "diet sau kajio gold", "kaijo"],
    "khai-hoang-g63": ["g63"],
    "khai-hoang-q7": ["q7"],
    "khai-hoang-q10": ["q10"],
    "khai-hoang-p7": ["p7"],
    "komulunx-80wg": ["diet nam komulunx", "tru nam komulunx 80wg", "komulunx"],
    "koto-240sc": ["diet sau koto", "tru sau koto 240sc", "koto"],
    "koto-240sc-gold": ["diet sau koto gold", "tru sau koto 240sc gold", "koto gold", "koto"],
    "kyodo-25sc": ["diet sau kyodo", "tru sau kyodo 25sc", "kyodo"],
    "kyodo-50wp": ["diet nam kyodo", "tru nam kyodo 50wp", "kyodo"],
    "lyrhoxini": ["diet sau lyrhoxini", "tru sau lyrhoxini", "lyrhoxini"],
    "m8-sing": ["diet sau m8 sing", "tru sau m8 sing", "m8", "m8sing"],
    "matscot-500sp": ["diet sau matscot", "tru sau matscot 500sp", "matscot"],
    "newfosinate": ["diet co fosinate moi", "diet co newfosinate", "newfosinate"],
    "newtongard-75": ["diet nam newtongard", "tru nam newtongard 75", "newtongard"],
    "niko-72wp": ["niko 72wp", "niko 72wp", "tru sau niko 72wp", "niko"],
    "oosaka-700wp": ["oosaka 700wp", "oosaka 700wp", "tru sau oosaka 700wp", "oosaka"],
    "phuong-hoang-lua": ["phuong hoang"],
    "recxona-350SC": ["recxona 350sc", "recxona 350sc", "recxona"],
    "scrotlan-80wp-m45-an": ["scrotlan 80wp m45 an", "scrotlan 80wp m45 an", "scrotlan"],
    "snoil-delta-thuy-si": ["snoil delta thuy si", "snoil delta thuy si", "snoil", "oc", "tru oc", "diet oc", "tri oc"],
    "soccarb-80wg": ["soccarb 80wg", "soccarb 80wg", "soccarb"],
    "teen-super-350ec": ["teen super", "teensuper", "teen super 350ec"],
    "tosi-30wg": ["tosi 30wg", "tosi"],
    "trau-den-150sl": ["trau den 150sl", "trau den"],
    "trau-rung-2-0": ["trau rung 2 0", "trau rung"],
    "trau-rung-moi": ["trau rung moi", "trau rung"],
    "trau-vang-280": ["trau vang 280", "trau vang"],
    "trum-nam-benh": ["trum nam benh", "trum nam benh", "nam benh"],
    "trum-sau": ["trum sau", "trum sau"],
    "truong-doi-ky": ["truong doi ky", "truong doi ky"],
    "vua-imida": ["vua imida", "vua imida", "imida"],
    "vua-rep-sau-ray": ["vua rep sau ray", "vua rep", "vua sau ray"],
    "vua-sau": ["vua sau", "vua sau"],
    "vua-tri": ["vua tri", "vua tri"],
    "zeroanvil": ["zeroanvil", "zeroanvil"],
    "Afenzole": ["afenzole", "afenzole"],
    "Oosaka": ["oosaka", "oosaka", "oc", "tru oc", "diet oc", "tri oc"],
    "Recxona-35WG": ["recxona 35wg", "recxona"],
    "Sinapyram": ["sinapyram", "sinapyram"],
    "abamectin-3-6-duc": ["abamectin 3 6 duc", "diet sau abamectin", "abamectin", "abamectin"],
    "abinsec": ["abinsec", "abinsec"],
    "abinsec-1-8ec": ["abinsec 1 8ec", "abinsec 1.8ec", "abinsec 1.8", "abinsec 1.8ec ", "abinsec"],
    "abinsec-emaben-sau-1-8ec": ["abinsec emaben sau 1 8ec", "abinsec", "emaben"],
    "abinsec-oxatin-1-8ec": ["abinsec oxatin 1 8ec", "oxatin", "abinsec", "abinsec oxatin", "abinsec oxatin"],
    "abinsec-sieu-diet-nhen": ["abinsec diet nhen sieu toc", "sieu diet nhen", "abinsec nhen", "abinsec"],
    "aco-one-40ec": ["aco one 40ec", "aco one", "acoone"],
    "aco-one400ec": ["aco one 400ec", "aco one", "acoone"],
    "afenzole-325sc": ["afenzole 325sc", "afenzole", "afenzole 325sc", "afenzole"],
    "amesip-80wp": ["amesip 80wp", "amesip", "amesip"],
    "amkamec-3-6-ec": ["amkamec 3 6 ec", "amkamec", "amkamec 3.6ec", "amkamec"],
    "atomin": ["atomin", "atomin"],
    "atomin-15wp": ["atomin 15wp", "atomin", "atomin15wp"],
    "azin-rio-45sc": ["azin rio 45sc", "azinrio", "azin", "rio"],
    "bac-si-nam-benh-nekko-69wp": ["bac si nam benh nekko 69wp", "nekko"],
    "bacillus-suv": ["vi khuan bacillus suv", "vi khuan bacillus suv", "bacillus"],
    "bam-dinh-bmc": ["bam dinh bmc", "bam dinh"],
    "bao-den": ["bao den"],
    "basuzin-1gb": ["diet co basuzin", "basuzin"],
    "bilu": ["diet sau bilu"],
    "binhfos-50ec": ["diet sau binhfos 50ec", "binhfoc 500ec", "binhfos", "binfos", "bin fos"],
    "binhfos-50ec-vua-ray-rep": ["diet ray rep binhfos 50ec", "vua ray rep", "binhfos", "binfos", "bin fos"],
    "binhfos-50ec-vua-sau": ["diet sau binhfos 50ec vua", "vua sau", "binhfos", "binfos", "bin fos"],
    "binhfos-anh-hung-sau": ["diet sau binhfos anh hung", "anh hung sau", "binhfos", "binfos", "bin fos"],
    "binhtox-3-8ec": ["diet sau binhtox 3 8ec", "binhtox", "binh tox", "bintox", "bin tox"],
    "binhtox-3-8ec-gold": ["diet sau binhtox 3 8ec vang", "binhtox-gold", "binhtox gold", "binhtox", "binh tox", "bintox", "bin tox"],
    "binhtox-gold": ["diet sau binhtox vang", "binhtox-gold", "binhtox gold", "binhtox", "binh tox", "bintox", "bin tox"],
    "biperin-100ec": ["diet sau biperin 100ec", "biperin"],
    "bipimai": ["diet sau bipimai", "bipimai", "bi pi"],
    "bipimai-150ec": ["diet sau bipimai 150ec", "bipimai", "bipimai 150ec", "bi pi"],
    "bisomin-2sl": ["diet sau bisomin 2sl", "bisomin"],
    "bisomin-6wp": ["diet sau bisomin 6wp", "bisomin"],
    "bn-fosthi-10gr": ["diet sau bn fosthi 10gr", "fosthi", "-fosthi"],
    "bn-meta-18gr": ["diet sau bn meta 18gr", "meta 18gr", "meta"],
    "bpalatox-100ec": ["diet sau bpalatox 100ec", "bpalatox"],
    "bpsaco-500ec": ["diet sau bpsaco 500ec", "bpsaco"],
    "bretil-super-300ec": ["diet sau bretil super 300ec", "bretil"],
    "buti": ["diet sau buti", "buti"],
    "buti-43sc": ["diet sau buti 43sc", "buti"],
    "butti-43sc-anh-hung-nhen": ["diet nhen butti 43sc anh hung", "anh hung nhen", "buti"],
    "byphan-800wp": ["diet sau byphan 800wp", "byphan"],
    "chessin-600wp": ["diet sau chessin 600wp", "chessin", "2 hoat chat", "hai hoat chat"],
    "chitin-daphamec-3-6": ["diet sau chitin daphamec 3 6", "chitin daphamec", "chitin", "chitin 3.6", "daphamec"],
    "chlorfena-240sc": ["diet sau chlorfena 240sc", "chlorfena"],
    "chowon-550sl": ["diet sau chowon 550sl than duoc", "chowon", "than duoc"],
    "cocosieu-15-5-wp": ["diet sau cocosieu 15 5 wp", "cocosieu"],
    "cocosieu-15-5wp": ["diet sau cocosieu 15 5 wp", "cocosieu"],
    "cymkill-25ec": ["diet sau cymkill 25ec", "cymkill"],
    "daphamec-5-0ec": ["diet sau daphamec 5 0ec", "daphamec"],
    "daphamec-5ec": ["diet sau daphamec 5ec", "daphamec"],
    "diet-nhen-sam-set": ["diet nhen sam set", "diet nhen sam set", "diet nhen", "sam set"],
    "dolping-40ec": ["diet sau dolping 40ec", "dolping"],
    "downy": ["diet benh downy", "downy"],
    "downy-650wp": ["diet benh downy 650wp", "downy"],
    "dp-avo": ["dp avo", "dp avo", "dp-avo", "avo"],
    "durong-800wp": ["durong 800wp", "durong", "durong-800wp"],
    "dusan-240ec": ["dusan 240ec", "dusan", "dusan-240ec"],
    "ecudor-22-4sc": ["ecudor 22 4sc", "ecudor"],
    "emoil-99ec-spray": ["em oil 99ec", "Petrolium", "em oil", "em-oil", "oil", "emoil"],
    "emycin-4wp": ["emycin 4wp", "emycin", "2 hoat chat", "hai hoat chat"],
    "exami": ["exami", "exami"],
    "exami-20wg": ["exami 20wg", "exami"],
    "exami-20wg-ly-tieu-long": ["exami 20wg ly tieu long", "exami", "ly tieu long"],
    "faptank": ["faptank", "faptank"],
    "faquatrio-20sl": ["faquatrio 20sl", "faquatrio"],
    "forcin-50ec": ["forcin 50ec", "forcin"],
    "forgon-40ec": ["forgon 40ec", "forgon"],
    "forsan-60ec": ["forsan 60ec", "forsan"],
    "forsan-horisan": ["forsan horisan", "forsan horisan", "horisan", "forsan"],
    "fortac-5ec": ["fortac 5ec", "fortac"],
    "fortazeb-72wp": ["fortazeb 72wp", "fortazeb"],
    "forthane-80wp": ["forthane 80wp", "forthane", "forthan"],
    "forwarat-0-005-wax-block": ["forwarat 0 005 wax block", "forwarat"],
    "forzate-20ec": ["forzate 20ec", "forzate"],
    "forzate-20ew": ["forzate 20ew", "forzate"],
    "fuji-boss-30sc": ["fuji boss 30sc", "fujiboss", "fuji boss", "boss", "fuji", "2 hoat chat", "hai hoat chat"],
    "fujiboss-30sc": ["fuji boss 30sc", "fujiboss", "fuji boss", "boss", "fuji", "2 hoat chat", "hai hoat chat"],
    "fullkill-50ec": ["fullkill 50ec", "fullkill", "full kill", "full-kill", "kill"],
    "gangter-300ec": ["gangter 300ec", "gangter"],
    "gardona-250sl": ["gardona 250sl", "gardona"],
    "giao-su-sau": ["giao su sau", "giao su sau"],
    "giaosu-co": ["giao su co", "giao su co"],
    "gibberellic-acid": ["axit gibberelic", "gibberellic"],
    "gly-888": ["gly 888", "gly", "888", "gly 888", "gly-888"],
    "gone-super": ["gone super", "gone super", "gon super", "gone super",],
    "gone-super-350ec": ["gone super 350ec", "gone super", "gone super 350ec", "gon super", "gone super",],
    "gorop-500ec": ["gorop 500ec", "gorop"],
    "gussi-bastar-200sl": ["gussi bastar 200sl", "gussi bastar", "gussi", "bastar"],
    "gussi-bastar-200sl-dac": ["gussi bastar 200sl dac", "gussi bastar 200sl dac", "gussi bastar", "gussi", "bastar"],
    "haihamec-3-6e": ["haihamec 3 6e", "haihamec", "haihamec3.6ec", "haihamec3.6"],
    "haihamec-3-6ec": ["haihamec 3 6ec", "haihamec", "haihamec3.6ec", "haihamec3.6"],
    "hariwon-30sl": ["hariwon 30sl", "hariwon", "hariwon30sl", "hariwon-30sl"],
    "haruko-5sc": ["haruko 5sc", "tru sau haruko 5sc", "haruko", "haruko-5sc", "haruko5sc"],
    "hello-fungi-400": ["bao ve nam hello fungi 400", "diet nam hello fungi 400", "hello-fungi", "hello fungi", "hello", "fungi"],
    "hexa": ["hexa", "hexa", "hexa"],
    "hoanganhvil-50sc": ["hoang anh vil 50sc", "hoang anh vil 50sc", "hoanganhvil"],
    "hong-ha-nhi": ["hong ha nhi", "hong ha nhi", "hong ha nhi"],
    "hosu-10sc": ["ho su 10sc", "ho su 10sc", "hosu10sc", "hosu"],
    "ic-top-28-1sc": ["ic top 28 1sc", "ic top 28 1sc", "ictop", "ic-top", "ic top"],
    "ic-top-28-1sc-boocdor": ["ic top 28 1sc boocdor", "ic top 28 1sc boocdor", "ictop", "ic-top", "ic top"],
    "igro-240sc": ["igro 240sc", "igro 240sc", "igro", "igro240sc"],
    "igro-240sc-ohayo": ["igro 240sc ohayo", "igro 240sc ohayo", "igro", "igro240sc"],
    "igro-240sc-xpro": ["igro 240sc xpro", "igro 240sc xpro","igro", "igro240sc"],
    "igro-ohayo-240sc": ["igro ohayo 240sc", "igro ohayo 240sc", "igro", "igro240sc"],
    "igro-xpro": ["igro xpro", "igro xpro", "igro", "igro240sc"],
    "inmanda-100wp": ["inmanda 100wp", "inmanda 100wp", "inmanda", "inmanda", "inmanda"],
    "japasa-50ec": ["japasa 50ec", "japasa 50ec", "japasa"],
    "jinhe-barass-0-01sl": ["jinhe barass 0 01sl", "jinhe barass 0 01sl", "jinhe-barass", "jinhe barass", "jinhe", "barass"],
    "jinhe-brass-0-01sl": ["jinhe brass 0 01sl", "jinhe brass 0 01sl"],
    "kajio-1gr": ["kajio 1gr", "kajio 1gr", "kajio 1gr", "kajio"],
    "kajio-1gr-alpha-anh-hung-sung": ["kajio 1gr alpha anh hung sung", "kajio 1gr alpha anh hung sung", "kajio alpha", "anh hung sung", "kajio-alpha"],
    "kajio-1gr-gold": ["kajio 1gr gold", "kajio 1gr gold", "kajio 1gr gold", "kajio gold", "kajio"],
    "kajio-5ec": ["kajio 5ec", "kajio 5ec", "kajio", "kajio-5ec", "kajio5ec"],
    "kajio-5ec-g9-thanh-sau": ["kajio 5ec g9 thanh sau", "kajio 5ec g9 thanh sau", "kajio g9", "kajio-g9", "kajio 5ec", "kajio", "kajio-5ec", "kajio thanh sau"],
    "kajio-5wg": ["kajio 5wg", "kajio 5wg", "kajio", "kajio-5wg"],
    "kasuhan-4wp": ["kasuhan 4wp", "kasuhan 4wp", "kasuhan 4wp", "kasuhan-4wp", "kasuhan"],
    "kenbast-15sl": ["kenbast 15sl", "kenbast", "kenbast-15sl"],
    "khongray-54wp": ["khongray 54wp", "khongray 54wp", "khongray", "khongray-54wp", "2 hoat chat", "hai hoat chat"],
    "king-cide-japan-460sc": ["king cide japan 460sc", "king cide japan 460sc", "king cide japan 460sc", "king cide japan", "king cide", "kingcide", "king-cide", "3 hoat chat", "ba hoat chat"],
    "king-kha-1ec": ["king kha 1ec", "king kha 1ec", "king kha 1ec", "king kha", "kinh kha", "kingkha"],
    "koto-240sc": ["koto 240sc", "koto 240sc", "koto"],
    "koto-gold-240sc": ["koto gold 240sc", "koto gold 240sc", "koto", "koto-gold", "koto gold", "koto-gold"],
    "kyodo": ["kyodo", "kyodo", "kyodo"],
    "kyodo-25sc": ["kyodo 25sc", "kyodo 25sc", "kyodo"],
    "kyodo-25sc-gold": ["kyodo 25sc gold", "kyodo 25sc gold", "kyodo", "kyodo-gold", "kyodo gold"],
    "kyodo-50wp": ["kyodo 50wp", "kyodo 50wp", "kyodo"],
    "lac-da": ["lac da", "lac da", "lac da"],
    "lama-50ec": ["lama 50ec", "lama 50ec", "lama"],
    "lao-ton-108ec": ["lao ton 108 ec", "lao ton 108 ec", "lao ton"],
    "laoton-108ec": ["lao ton 108 ec", "lao ton 108 ec", "lao ton"],
    "ledan-4gr": ["ledan 4 gr", "ledan 4 gr", "ledan", "ledan 4gr", "ledan 4g"],
    "ledan-95sp": ["ledan 95 sp", "ledan 95 sp", "ledan", "ledan 95sp", "ledan 95"],
    "lekima": ["lekima", "lekima"],
    "lekima-100ec": ["lekima 100 ec", "lekima 100 ec", "lekima", "lekima"],
    "lufenuron-5ec": ["lufenuron 5 ec", "diet sau lufenuron", "lufenuron 5 ec", "lufenuron"],
    "many-800wp": ["many 800 wp", "many 800 wp", "many"],
    "maruka-5ec": ["maruka 5 ec", "maruka 5 ec", "maruka"],
    "matscot": ["matscot", "matscot"],
    "matscot-50sp": ["matscot", "matscot 50 sp", "matscot 50 sp"],
    "matscot-50sp-ech-com": ["matscot", "matscot 50 sp ech com", "matscot 50 sp ech com"],
    "mekongvil-5sc": ["mekongvil", "mekongvil 5 sc", "mekongvil 5 sc"],
    "mi-stop-350sc": ["mistop", "mi-stop", "mi stop 350 sc", "mi stop 350 sc"],
    "million-50wg": ["million", "million 50 wg", "million 50 wg"],
    "miriphos-1gb": ["miriphos", "miriphos 1 gb", "miriphos 1 gb"],
    "misung-15sc": ["misung", "misung 15 sc", "misung 15 sc"],
    "mitop-one-390sc": ["mitop-one", "mitop one", "mitop", "mitop one 390sc", "3 hoat chat", "ba hoat chat"],
    "modusa-960ec": ["modusa", "modusa 960 ec", "modusa 960 ec"],
    "modusa-960ec-gold": ["modusa", "modusa 960 ec gold", "modusa 960 ec gold"],
    "nakano-50wp": ["nakano", "nakano 50 wp", "nakano 50 wp"],
    "napoleon-fortazeb-72wb": ["napoleon", "fortazeb", "napoleon-fortazeb", "napoleon fortazeb 72 wb", "napoleon fortazeb 72 wb"],
    "naticur": ["naticur", "naticur", "2 hoat chat", "hai hoat chat"],
    "nekko-69wp": ["nekko 69 wp", "nekko 69 wp", "nekko"],
    "newfosinate-150sl": ["newfosinate 150 sl", "newfosinate 150 sl", "newfosinate"],
    "nhen-kim-cuong": ["kim cuong", "diet nhen kim cuong"],
    "nhen-do": ["nhen do"],
    "nhen-gie": ["nhen gie"],
    "nhen-long-nhung": ["nhen long nhung"],
    "nhen-trang": ["nhen trang"],
    "nhen-vang": ["nhen vang"],
    "niko": ["niko", "niko"],
    "niko-72wp": ["niko 72 wp", "niko 72 wp", "niko"],
    "nofara": ["nofara", "nofara"],
    "nofara-350sc": ["nofara 350 sc", "nofara 350 sc", "nofara"],
    "nofara-35wg": ["nofara 35 wg", "nofara 35 wg", "nofara"],
    "oc-15gr": ["oc 15 gr", "oc 15 gr", "oc 15gr", "oc", "tru oc", "diet oc", "tri oc"],
    "oc-ly-tieu-long-18gr": ["oc ly tieu long 18 gr", "oc ly tieu long 18 gr", "oc ly tieu long", "ly tieu long 18gr", "oc", "tru oc", "diet oc", "tri oc"],
    "ohayo-100sc": ["ohayo 100 sc", "ohayo 100 sc", "ohayo 100sc", "ohayo"],
    "ohayo-240sc": ["ohayo 240 sc", "ohayo 240 sc", "ohayo 240sc", "ohayo"],
    "onehope": ["onehope", "onehope", "one hope"],
    "onehope-480sl": ["onehope 480sl" ,"onehope 480 sl", "onehope 480 sl", "onehope", "one hope"],
    "oosaka-700wp": ["oosaka", "oosaka 700 wp", "oosaka 700 wp"],
    "oscare-100wp": ["oosaka", "oscare 100 wp", "oscare 100 wp", "oscare"],
    "oscare-600wg": ["oosaka", "oscare 600 wg", "oscare 600 wg", "oscare"],
    "oxatin": ["oxatin", "oxatin"],
    "oxine-copper": ["dong oxin", "dong oxin", "oxine", "copper"],
    "panda-4gr": ["phan panda", "phan panda 4gr", "panda 4gr", "panda-4gr", "panda"],
    "parato-200sl": ["parato", "parato 200sl", "parato"],
    "parato-than-lua": ["parato chong than lua", "parato than lua", "parato"],
    "paskin-250": ["paskin", "paskin 250", "paskin"],
    "phonix-dragon-20ec": ["phonix dragon", "phonix dragon 20ec","phonix dragon", "phonix", "dragon"],
    "phuong-hoang-lua": ["phuong hoang", "phuong hoang lua", "phuong hoang lua", "phuong hoang"],
    "pilot-15ab": ["pilot", "pilot 15ab", "pilot", "oc", "tru oc", "diet oc", "tri oc"],
    "pim-pim-75wp": ["pim pim", "pim pim 75wp", "pimpim", "pim-pim"],
    "probicol-200wp": ["probicol", "probicol 200wp", "probicol", "propicol", "2 hoat chat", "hai hoat chat"],
    "prochloraz-manganese-50-wp": ["prochloraz mangan 50wp", "prochloraz manganese", "prochloraz"],
    "pyrolax-250ec": ["pyrolax", "pyrolax 250ec", "pyrolax"],
    "ram-te-thien": ["ram te", "ram te thien", "te thien"],
    "raynanusa-400wp": ["raynanusa", "raynanusa 400wp", "raynanusa"],
    "riceup-300ec": ["riceup", "riceup 300ec", "riceup"],
    "sam-san-2-5sc": ["sam san", "sam san 2 5sc", "sam san", "sam-san"],
    "sanbang-30sc": ["sanbang", "sanbang 30sc", "sanbang", "san bang", "sangbang", "sang bang"],
    "santoso-100sc": ["santoso", "santoso 100sc", "santoso"],
    "scorcarb-80wg": ["scorcarb", "scorcarb 80wg", "scorcarb"],
    "scortlan": ["scortlan", "scortlan"],
    "scortlan-80wp": ["scortlan 80wp", "scortlan"],
    "setis-34sc": ["setis", "setis 34sc", "setis"],
    "setis-giao-su-nhen-34sc": ["setis giao su nhen", "setis giao su nhen 34sc", "giao su nhen", "setis"],
    "sha-chong-jing": ["sha chong jing", "sha chong jing", "sha chong"],
    "sha-chong-jing-95wp": ["sha chong jing 95wp", "sha chong jing", "sha chong"],
    "shina-18sl": ["shina", "shina 18sl", "shina"],
    "shinawa-400ec": ["shinawa", "shinawa 400ec", "shinawa"],
    "shonam-500sc": ["shonam", "shonam 500sc", "shonam"],
    "showbiz": ["showbiz", "showbiz"],
    "showbiz-16sc": ["showbiz 16sc", "showbiz"],
    "sieu-bam-dinh": ["bam dinh", "sieu bam dinh", "sieu bam dinh", "bam dinh"],
    "sieu-diet-chuot": ["diet chuot", "sieu diet chuot", "diet chuot"],
    "sieu-diet-mam": ["diet mam", "sieu diet mam", "diet mam"],
    "sieu-diet-nhen": ["diet nhen", "sieu diet nhen", "diet nhen"],
    "sieu-diet-sau": ["diet sau", "sieu diet sau", "diet-sau"],
    "sinapy-ram": ["sinapy", "sinapy ram", "sinapy", "sinapyram", "ram"],
    "sinapyram-80wg": ["sinapyram 80wg", "sinapy", "sinapyram", "ram"],
    "somethrin-10ec": ["somethrin", "somethrin 10ec", "somethrin"],
    "su-tu-do": ["su tu do", "su tu do"],
    "suparep-22-4sc": ["suparep 22 4sc", "suparep", "suparep"],
    "suparep-400wp": ["suparep 400wp", "suparep"],
    "supermario-70sc": ["supermario 70sc", "supermario", "2 hoat chat", "hai hoat chat"],
    "suria-10gr": ["suria 10gr", "suria"],
    "suron-800wp": ["suron 800wp", "suron"],
    "takiwa-22sc": ["takiwa 22sc", "takiwa"],
    "tamatras": ["tamatras", "tamatras", "spirotetramat tolfenpyrad", "spirotetramat", "tolfenpyrad"],
    "tamiko-50ec": ["tamiko 50ec", "tamiko"],
    "tatsu-25wp": ["tatsu 25wp", "tatsu"],
    "tembo-8od-vua-co-ngo": ["tembo 8od vua co ngo", "tembo", "vua co ngo", "8od"],
    "thalonil-75wp": ["thalonil 75wp", "thalonil"],
    "thuoc-bilu": ["bilu"],
    "thuoc-chuot-forwarat-0-005-wax-block": ["chuot forwarat 0 005 wax block"],
    "tomi": ["tomi", "tomi"],
    "tomi-5ec": ["tomi 5ec", "tomi"],
    "topmesi-40sc": ["topmesi 40sc", "topmesi"],
    "topxapy-30sc": ["topxapy 30sc", "topxapy"],
    "topxim-pro-30sc": ["topxim pro 30sc", "topxim", "topximpro"],
    "toshiro-10ec": ["toshiro 10ec", "toshiro"],
    "tosi": ["tosi", "tosi"],
    "tosi-30wg": ["tosi 30wg", "tosi"],
    "trau-den-150": ["trau den 150", "trau den"],
    "trau-rung-2.0": ["trau rung 2 0", "trau rung", "trau rung2.0"],
    "trau-rung-moi": ["trau rung moi", "trau rung"],
    "trau-vang": ["trau vang", "trau vang"],
    "trinong-50wp": ["trinong 50wp", "trinong"],
    "trum-chich-hut-tri": ["trum chich hut tri", "trum chich hut", "chich hut"],
    "truong-vo-ky": ["truong vo ky", "truong vo ky", "yosky"],
    "uchong-40ec": ["uchong 40ec", "uchong"],
    "vamco-480sl": ["vamco 480sl", "vam co", "vamco", "dktazole", "dktazone"],
    "vam-co-dktazole-480sl": ["vam co dktazole 480sl", "vam co", "dktazole", "vamco"],
    "vet-xanh": ["vet xanh", "vetxanh", "vet-xanh", "vet xanh"],
    "voi-rung": ["voi rung", "voi rung"],
    "voi-thai-3-6ec-gold": ["voi thai 3 6ec gold", "voi thai"],
    "vua-co-ngo": ["vua co ngo", "vua co ngo"],
    "vua-lua-chay-khai-hoang-malaysia": ["vua lua chay khai hoang malaysia", "vua lua chay", "khai hoang malaysia", "malaysia"],
    "vua-mida-phuong-hoang-lua": ["vua mida phuong hoang lua", "vua mida", "phuong hoang lua", "mida"],
    "wusso": ["wusso", "wusso"],
    "xie-xie-200wp": ["xie xie 200wp", "xie xie", "xie-xie", "xiexie", "xie"],
    "xiexie-200wp-anh-hung-khuan": ["diet khuan xiexie 200wp anh hung", "anh hung khuan", "xie xie", "xie-xie", "xiexie", "xie"],
    "yosky": ["yosky", "yosky, 10sl"],
    "yosky-10sl-khai-hoang-p7": ["yosky 10sl khai hoang p7", "yosky"],
    "zigen": ["zigen", "zigen"],
    "zigen-15sc": ["zigen 15sc", "zigen"],
    "zigen-super": ["zigen super", "zigen"],
    "zigen-super-15sc": ["zigen super 15sc", "zigen"],
    "zigen-xpro": ["zigen xpro", "zigen"],
}

BRAND_ALIASES = {
    "bmc": ["san pham bmc", "cong ty bmc", "cua bmc"],
    "phuc-thinh": ["san pham phuc thinh", "cong ty phuc thinh", "cua phuc thinh"],
    "agrishop": ["san pham agrishop", "cong ty agrishop", "cua agrishop"],
    "delta": ["san pham delta", "cong ty delta", "cua delta"],
}

MECHANISMS_ALIASES = {
    "tiep-xuc-luu-dan-manh": ["tiep xuc va luu dan manh", "tiep xuc va luu dan nao manh", " tiep xuc luu dan nao manh", "luu dan manh, tiep xuc", "tiep xuc manh", "tiep xuc nao manh", "luu dan manh", "luu dan nao manh"],
    "tiep-xuc-luu-dan": ["tiep xuc va luu dan", "tiep xuc va luu dan nao", "tiep xuc luu dan nao", "tiep xuc, luu dan", "luu dan, tiep xuc", "tiep xuc", "tiep suc", "tiep xuc nao", "tiep suc nao", "luu dan", "lu dan", "luu dan nao", "lu dan nao"],
    "luu-dan-manh": ["luu dan manh", "luu dan nao manh"],
    "tiep-xuc-manh": ["tiep xuc manh", "tiep xuc nao manh"],
    "xong-hoi-manh": ["xong hoi manh", "xong hoi nao manh"],
    "luu-dan": ["luu dan", "lu dan", "luu dan nao", "lu dan nao"],
    "tiep-xuc": ["tiep xuc", "tiep suc", "tiep xuc nao", "tiep suc nao"],
    "xong-hoi": ["xong hoi"],
    "co-chon-loc": ["co chon loc", "bao trum", "trum", "lua"],
    "khong-chon-loc": ["khong chon loc", "k chon loc"],
    "dac-tri-khuan": ["tri khuan", "dac tri vi khuan"],
    "dac-tri-rep": ["dac tri rep", "dac tri bo"],
    "dac-tri-tri": ["dac tri tri", "dac tri ray"],
    "dac-tri-sau": ["dac tri sau"],
    "dac-tri-ray": ["dac tri ray", "dac tri chich hut"],
    "dac-tri-nhen": ["dac tri nhen"],
    "dac-tri-dao-on": ["dac tri dao on"],
    "dac-tri-sau-to": ["dac tri sau to", "dac tri sau hanh"],
}

FORMULA_ALIASES = {
    "cong-thuc-ray-nau": [
        "cong thuc tru ray nau",
        "cong thuc diet ray nau",
        "cong thuc phong tru ray nau",
        "cong thuc tri ray nau",
        "cong thuc thuoc tru ray nau",
        "cong thuc thuoc diet ray nau",
        "cong thuc thuoc phong tru ray nau",
        "cong thuc thuoc tri ray nau",
    ],
    "cong-thuc-ray-lung-trang": [
        "cong thuc tru ray lung trang",
        "cong thuc diet ray lung trang",
        "cong thuc phong tru ray lung trang",
        "cong thuc tri ray lung trang",
        "cong thuc thuoc tru ray lung trang",
        "cong thuc thuoc diet ray lung trang",
        "cong thuc thuoc phong tru ray lung trang",
        "cong thuc thuoc tri ray lung trang",
    ],
    "cong-thuc-ray-xanh": [
        "cong thuc tru ray xanh",
        "cong thuc diet ray xanh",
        "cong thuc phong tru ray xanh",
        "cong thuc tri ray xanh",
        "cong thuc thuoc tru ray xanh",
        "cong thuc thuoc diet ray xanh",
        "cong thuc thuoc phong tru ray xanh",
        "cong thuc thuoc tri ray xanh",
    ],
    "cong-thuc-ray-chong-canh": [
        "cong thuc tru ray chong canh",
        "cong thuc diet ray chong canh",
        "cong thuc phong tru ray chong canh",
        "cong thuc tri ray chong canh",
        "cong thuc thuoc tru ray chong canh",
        "cong thuc thuoc diet ray chong canh",
        "cong thuc thuoc phong tru ray chong canh",
        "cong thuc thuoc tri ray chong canh",
    ],
    "cong-thuc-ray-phan": [
        "cong thuc tru ray phan",
        "cong thuc diet ray phan",
        "cong thuc phong tru ray phan",
        "cong thuc tri ray phan",
        "cong thuc thuoc tru ray phan",
        "cong thuc thuoc diet ray phan",
        "cong thuc thuoc phong tru ray phan",
        "cong thuc thuoc tri ray phan",
    ],
    "cong-thuc-ray-mem": [
        "cong thuc tru ray mem",
        "cong thuc diet ray mem",
        "cong thuc phong tru ray mem",
        "cong thuc tri ray mem",
        "cong thuc thuoc tru ray mem",
        "cong thuc thuoc diet ray mem",
        "cong thuc thuoc phong tru ray mem",
        "cong thuc thuoc tri ray mem",
    ],
    "cong-thuc-bo-phan-trang": [
        "cong thuc tru bo phan trang",
        "cong thuc diet bo phan trang",
        "cong thuc phong tru bo phan trang",
        "cong thuc tri bo phan trang",
        "cong thuc thuoc tru bo phan trang",
        "cong thuc thuoc diet bo phan trang",
        "cong thuc thuoc phong tru bo phan trang",
        "cong thuc thuoc tri bo phan trang",
    ],
    "cong-thuc-bo-xit": [
        "cong thuc tru bo xit",
        "cong thuc diet bo xit",
        "cong thuc phong tru bo xit",
        "cong thuc tri bo xit",
        "cong thuc thuoc tru bo xit",
        "cong thuc thuoc diet bo xit",
        "cong thuc thuoc phong tru bo xit",
        "cong thuc thuoc tri bo xit",
    ],
    "cong-thuc-rep-sap": [
        "cong thuc tru rep sap",
        "cong thuc diet rep sap",
        "cong thuc phong tru rep sap",
        "cong thuc tri rep sap",
        "cong thuc thuoc tru rep sap",
        "cong thuc thuoc diet rep sap",
        "cong thuc thuoc phong tru rep sap",
        "cong thuc thuoc tri rep sap",
    ],
    "cong-thuc-rep-vay": [
        "cong thuc tru rep vay",
        "cong thuc diet rep vay",
        "cong thuc phong tru rep vay",
        "cong thuc tri rep vay",
        "cong thuc thuoc tru rep vay",
        "cong thuc thuoc diet rep vay",
        "cong thuc thuoc phong tru rep vay",
        "cong thuc thuoc tri rep vay",
    ],
    "cong-thuc-bo-tri": [ 
        "cong thuc tru bo tri", 
        "cong thuc diet bo tri", 
        "cong thuc phong tru bo tri", 
        "cong thuc tri bo tri", 
        "cong thuc thuoc tru bo tri", 
        "cong thuc thuoc diet bo tri", 
        "cong thuc thuoc phong tru bo tri", 
        "cong thuc thuoc tri bo tri", 
    ],
    "cong-thuc-sau-to": [
        "cong thuc tru sau to",
        "cong thuc diet sau to",
        "cong thuc phong tru sau to",
        "cong thuc tri sau to",
        "cong thuc thuoc tru sau to",
        "cong thuc thuoc diet sau to",
        "cong thuc thuoc phong tru sau to",
        "cong thuc thuoc tri sau to",
    ],
    "cong-thuc-sau-hanh": [
        "cong thuc tru sau hanh",
        "cong thuc diet sau hanh",
        "cong thuc phong tru sau hanh",
        "cong thuc tri sau hanh",
        "cong thuc thuoc tru sau hanh",
        "cong thuc thuoc diet sau hanh",
        "cong thuc thuoc phong tru sau hanh",
        "cong thuc thuoc tri sau hanh",
    ],
    "cong-thuc-bo-nhay": [
        "cong thuc tru bo nhay",
        "cong thuc diet bo nhay",
        "cong thuc phong tru bo nhay",
        "cong thuc tri bo nhay",
        "cong thuc thuoc tru bo nhay",
        "cong thuc thuoc diet bo nhay",
        "cong thuc thuoc phong tru bo nhay",
        "cong thuc thuoc tri bo nhay",
    ],
    "cong-thuc-sau-cuon-la": [
        "cong thuc tru sau cuon la",
        "cong thuc diet sau cuon la",
        "cong thuc phong tru sau cuon la",
        "cong thuc tri sau cuon la",
        "cong thuc thuoc tru sau cuon la",
        "cong thuc thuoc diet sau cuon la",
        "cong thuc thuoc phong tru sau cuon la",
        "cong thuc thuoc tri sau cuon la",
    ],
    "cong-thuc-sau-duc-than": [
        "cong thuc tru sau duc than",
        "cong thuc diet sau duc than",
        "cong thuc phong tru sau duc than",
        "cong thuc tri sau duc than",
        "cong thuc thuoc tru sau duc than",
        "cong thuc thuoc diet sau duc than",
        "cong thuc thuoc phong tru sau duc than",
        "cong thuc thuoc tri sau duc than",
    ],
    "cong-thuc-sau-duc-than": [
        "cong thuc tru sau duc than",
        "cong thuc diet sau duc than",
        "cong thuc phong tru sau duc than",
        "cong thuc tri sau duc than",
        "cong thuc thuoc tru sau duc than",
        "cong thuc thuoc diet sau duc than",
        "cong thuc thuoc phong tru sau duc than",
        "cong thuc thuoc tri sau duc than",
    ],
    "cong-thuc-sau-duc-qua": [
        "cong thuc tru sau duc qua",
        "cong thuc diet sau duc qua",
        "cong thuc phong tru sau duc qua",
        "cong thuc tri sau duc qua",
        "cong thuc thuoc tru sau duc qua",
        "cong thuc thuoc diet sau duc qua",
        "cong thuc thuoc phong tru sau duc qua",
        "cong thuc thuoc tri sau duc qua",
    ],
    "cong-thuc-sau-ve-bua": [
        "cong thuc tru sau ve bua",
        "cong thuc diet sau ve bua",
        "cong thuc phong tru sau ve bua",
        "cong thuc tri sau ve bua",
        "cong thuc thuoc tru sau ve bua",
        "cong thuc thuoc diet sau ve bua",
        "cong thuc thuoc phong tru sau ve bua",
        "cong thuc thuoc tri sau ve bua",
    ],
    "cong-thuc-sung": [
        "cong thuc tru sung",
        "cong thuc diet sung",
        "cong thuc phong tru sung",
        "cong thuc tri sung",
        "cong thuc thuoc tru sung",
        "cong thuc thuoc diet sung",
        "cong thuc thuoc phong tru sung",
        "cong thuc thuoc tri sung",
    ],
    "cong-thuc-bo-ha": [
        "cong thuc tru bo ha",
        "cong thuc diet bo ha",
        "cong thuc phong tru bo ha",
        "cong thuc tri bo ha",
        "cong thuc thuoc tru bo ha",
        "cong thuc thuoc diet bo ha",
        "cong thuc thuoc phong tru bo ha",
        "cong thuc thuoc tri bo ha",
    ],
    "cong-thuc-nhen": [
        "cong thuc tru nhen",
        "cong thuc diet nhen",
        "cong thuc phong tru nhen",
        "cong thuc tri nhen",
        "cong thuc thuoc tru nhen",
        "cong thuc thuoc diet nhen",
        "cong thuc thuoc phong tru nhen",
        "cong thuc thuoc tri nhen",
    ],
    "cong-thuc-nhen-khang-cao": [
        "cong thuc tru nhen khang cao",
        "cong thuc diet nhen khang cao",
        "cong thuc phong tru nhen khang cao",
        "cong thuc tri nhen khang cao",
        "cong thuc thuoc tru nhen khang cao",
        "cong thuc thuoc diet nhen khang cao",
        "cong thuc thuoc phong tru nhen khang cao",
        "cong thuc thuoc tri nhen khang cao",
    ],
    "cong-thuc-oc-buu-vang": [
        "cong thuc tru oc buu vang",
        "cong thuc diet oc buu vang",
        "cong thuc phong tru oc buu vang",
        "cong thuc tri oc buu vang",
        "cong thuc thuoc tru oc buu vang",
        "cong thuc thuoc diet oc buu vang",
        "cong thuc thuoc phong tru oc buu vang",
        "cong thuc thuoc tri oc buu vang",
    ],
    "cong-thuc-oc-sen": [
        "cong thuc tru oc sen",
        "cong thuc diet oc sen",
        "cong thuc phong tru oc sen",
        "cong thuc tri oc sen",
        "cong thuc thuoc tru oc sen",
        "cong thuc thuoc diet oc sen",
        "cong thuc thuoc phong tru oc sen",
        "cong thuc thuoc tri oc sen",
    ],
    "cong-thuc-oc-ma": [
        "cong thuc tru oc ma",
        "cong thuc diet oc ma",
        "cong thuc phong tru oc ma",
        "cong thuc tri oc ma",
        "cong thuc thuoc tru oc ma",
        "cong thuc thuoc diet oc ma",
        "cong thuc thuoc phong tru oc ma",
        "cong thuc thuoc tri oc ma",
    ],
    "cong-thuc-oc-nhot": [
        "cong thuc tru oc nhot",
        "cong thuc diet oc nhot",
        "cong thuc phong tru oc nhot",
        "cong thuc tri oc nhot",
        "cong thuc thuoc tru oc nhot",
        "cong thuc thuoc diet oc nhot",
        "cong thuc thuoc phong tru oc nhot",
        "cong thuc thuoc tri oc nhot",
    ],
}

FORMULATION_ALIASES = {
    "ab": ["ab"],
    "sc": ["sc"],
    "ec": ["ec"],
    "sl": ["sl"],
    "wp": ["wp"],
    "wg": ["wg"],
    "od": ["od"],
    "sp": ["sp"],
    "gb": ["gb", "ba moi", "oc rai", "rai goc"],
    "gr": ["gr", "rai goc", "rai goc"],
}

INTENT_ALIAS_GROUPS = {
    "formula": FORMULA_ALIASES,
    "mechanisms": MECHANISMS_ALIASES,
    "brand": BRAND_ALIASES,
    "product": PRODUCT_ALIASES,
    "chemical": CHEMICAL_ALIASES
}

# ===========================
# 4) MATCHING UTILITIES
# ===========================

def match_aliases(text: str, aliases: Dict[str, List[str]], normalizer: Callable[[str], str]) -> Set[str]:
    """
    Match theo word-boundary (khoảng trắng) sau khi normalize.
    """
    found: Set[str] = set()
    norm_text = f" {normalizer(text)} "

    for key, variants in aliases.items():
        for v in variants:
            alias = normalizer(v)
            if not alias:
                continue
            pattern = rf"(?:^|\s){re.escape(alias)}(?:\s|$)"
            if re.search(pattern, norm_text):
                found.add(key)
                break

    return found


# ===========================
# 6) UNIFIED KB INFERENCE (TARGET-FIRST + CROP-FALLBACK)
# ===========================

def infer_chemicals_from_kb(
    crops: Set[str],
    diseases: Set[str],
    pests: Set[str],
) -> Tuple[Set[str], str]:
    """
    Trả về (chemicals, mode)
    mode ∈ {"weed","pest","disease","crop","none"}

    Logic:
    1) weed -> match KB.weeds (lọc theo crop nếu có)
    2) pest -> match KB.pests (lọc theo crop nếu có)
    3) disease -> match KB.diseases (lọc theo crop nếu có)
    4) fallback crop-only -> match KB.crops
    """

    crops_n = normalize_set(crops, normalize_entity)
    diseases_n = normalize_set(diseases, normalize_entity)
    pests_n = normalize_set(pests, normalize_entity)

    def _filter_by_crop_if_any(kb_crops: Set[str]) -> bool:
        # nếu user có crop thì bắt buộc intersect; nếu không có crop thì không chặn
        return (not crops_n) or bool(crops_n.intersection(kb_crops))

    def _match_target(target: Set[str], kb_field: str) -> Set[str]:
        out = set()
        if not target:
            return out
        for chem, data in CHEMICAL_KB.items():
            kb_targets = set(data.get(kb_field, []))
            kb_crops = set(data.get("crops", []))
            if target.intersection(kb_targets) and _filter_by_crop_if_any(kb_crops):
                out.add(chem)
        return out

    # 2) Pest
    chems = _match_target(pests_n, "pests")
    if chems:
        return chems, "pest"

    # 3) Disease
    chems = _match_target(diseases_n, "diseases")
    if chems:
        return chems, "disease"

    # 4) Crop fallback (OLD behavior)
    if crops_n:
        out = set()
        for chem, data in CHEMICAL_KB.items():
            kb_crops = set(data.get("crops", []))
            if crops_n.intersection(kb_crops):
                out.add(chem)
        if out:
            return out, "crop"

    return set(), "none"


def filter_chemicals_by_formulation(chems: Set[str], forms: Set[str]) -> Set[str]:
    if not forms:
        return chems  # user không yêu cầu dạng → giữ nguyên

    forms_n = normalize_set(forms, normalize_entity)
    out = set()
    for c in chems:
        kb = CHEMICAL_KB.get(c, {})
        kb_forms = set(kb.get("formulation", []))
        # có ít nhất 1 dạng trùng
        if kb_forms.intersection(forms_n):
            out.add(c)
    return out


# ===========================
# 7) TAG EXTRACTION CORE
# ===========================

def extract_tags(norm_query_raw: str) -> Dict:
    """
    norm_query_raw: string đã normalize() (giữ + - /)
    """

    # Entity match: dùng normalize_entity để tránh mismatch do '-'
    crops = match_aliases(norm_query_raw, CROP_ALIASES, normalize_entity)
    diseases = match_aliases(norm_query_raw, DISEASE_ALIASES, normalize_entity)
    pests = match_aliases(norm_query_raw, PEST_ALIASES, normalize_entity)

    products = match_aliases(norm_query_raw, PRODUCT_ALIASES, normalize_entity)
    # brands = match_aliases(norm_query_raw, BRAND_ALIASES, normalize_entity)
    formulas = match_aliases(norm_query_raw, FORMULA_ALIASES, normalize_entity)
    forms = match_aliases(norm_query_raw, FORMULATION_ALIASES, normalize_entity)
    mechanisms = match_aliases(norm_query_raw, MECHANISMS_ALIASES, normalize_entity)

    # Chemical match: dùng normalize() để giữ tên có dấu '-'
    direct_chems = match_aliases(norm_query_raw, CHEMICAL_ALIASES, normalize)

    kb_chems, kb_mode = infer_chemicals_from_kb(crops, diseases, pests)

    # all chemicals: explicit + inferred
    all_chems = set(direct_chems).union(kb_chems)

    # FILTER theo formulation
    all_chems = filter_chemicals_by_formulation(all_chems, forms)

    # ======================
    # BUILD TAGS (must/any)
    # ======================

    must_tags: Set[str] = set()
    any_tags: Set[str] = set()

    # Crop/Pest/Weed: để ANY (noisy nhưng useful)
    for c in crops:
        any_tags.add(f"crop:{c}")
    for p in pests:
        any_tags.add(f"pest:{p}")

    # Disease thường khá "hard" -> MUST
    for d in diseases:
        any_tags.add(f"disease:{d}")

    # Product/Brand/Formulation/Formula: MUST
    for p in products:
        must_tags.add(f"product:{p}")
    # for b in brands:
    #     must_tags.add(f"brand:{b}")
    for f in forms:
        must_tags.add(f"formulation:{f}")
    for fm in formulas:
        must_tags.add(f"formula:{fm}")
    for mec in mechanisms:
        must_tags.add(f"mechanisms:{mec}")

    # Chemicals:
    # - direct_chems (người dùng nói thẳng) -> MUST
    # - kb_chems (suy diễn) -> ANY
    for chem in direct_chems:
        must_tags.add(f"chemical:{chem}")
    for chem in (all_chems - set(direct_chems)):
        any_tags.add(f"chemical:{chem}")

    return {
        "must": sorted(list(must_tags)),
        "any": sorted(list(any_tags)),
    }


# ===========================
# 8) MAIN PIPELINE
# ===========================

def tag_filter_pipeline(query: str) -> Dict:
    norm_raw = normalize(query)

    # 1) Ontology core
    tags = extract_tags(norm_raw)

    # MUST = ontology MUST + mechanisms MUST
    must_tags = set(tags["must"])

    # ANY = ontology ANY (crop/pest/weed + inferred chems) + fallback intent aliases
    detected_any = set(tags["any"])

    # fallback: thêm crop/pest/weed nếu match được (để tránh thiếu do extract_tags thay đổi)
    for c in match_aliases(norm_raw, CROP_ALIASES, normalize_entity):
        detected_any.add(f"crop:{c}")
    for p in match_aliases(norm_raw, PEST_ALIASES, normalize_entity):
        detected_any.add(f"pest:{p}")

    return {
        "query": query,
        "must": sorted(list(must_tags)),
        "any": sorted(list(detected_any)),
    }


# ===========================
# 9) TEST HARNESS
# ===========================

if __name__ == "__main__":
    tests = [
        "Trong 3 công thức trên, công thức nào là mạnh nhất?",
    ]

    for q in tests:
        print("\n==============================")
        print("QUERY:", q)
        print(json.dumps(tag_filter_pipeline(q), indent=2, ensure_ascii=False))