# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import string
from typing import Dict

__all__ = ["VOCABS"]


VOCABS: Dict[str, str] = {
    "digits": string.digits,
    "ascii_letters": string.ascii_letters,
    "punctuation": string.punctuation,
    "currency": "£€¥¢฿",
    "ancient_greek": "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ",
    "arabic_letters": "ءآأؤإئابةتثجحخدذرزسشصضطظعغـفقكلمنهوىي",
    "persian_letters": "پچڢڤگ",
    "arabic_digits": "٠١٢٣٤٥٦٧٨٩",
    "arabic_diacritics": "ًٌٍَُِّْ",
    "arabic_punctuation": "؟؛«»—",
    "hindi_letters": "अआइईउऊऋॠऌॡएऐओऔअंअःकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह",
    "hindi_digits": "०१२३४५६७८९",
    "hindi_punctuation": "।,?!:्ॐ॰॥॰",
    "bangla_letters": "অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ়ঽািীুূৃেৈোৌ্ৎংঃঁ",
    "bangla_digits": "০১২৩৪৫৬৭৮৯",
    "generic_cyrillic_letters": "абвгдежзийклмнопрстуфхцчшщьюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЮЯ",
}

VOCABS["latin"] = VOCABS["digits"] + VOCABS["ascii_letters"] + VOCABS["punctuation"]
VOCABS["english"] = VOCABS["latin"] + "°" + VOCABS["currency"]
VOCABS["legacy_french"] = VOCABS["latin"] + "°" + "àâéèêëîïôùûçÀÂÉÈËÎÏÔÙÛÇ" + VOCABS["currency"]
VOCABS["french"] = VOCABS["english"] + "àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ"
VOCABS["portuguese"] = VOCABS["english"] + "áàâãéêíïóôõúüçÁÀÂÃÉÊÍÏÓÔÕÚÜÇ"
VOCABS["spanish"] = VOCABS["english"] + "áéíóúüñÁÉÍÓÚÜÑ" + "¡¿"
VOCABS["italian"] = VOCABS["english"] + "àèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ"
VOCABS["german"] = VOCABS["english"] + "äöüßÄÖÜẞ"
VOCABS["arabic"] = (
    VOCABS["digits"]
    + VOCABS["arabic_digits"]
    + VOCABS["arabic_letters"]
    + VOCABS["persian_letters"]
    + VOCABS["arabic_diacritics"]
    + VOCABS["arabic_punctuation"]
    + VOCABS["punctuation"]
)
VOCABS["czech"] = VOCABS["english"] + "áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ"
VOCABS["polish"] = VOCABS["english"] + "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ"
VOCABS["dutch"] = VOCABS["english"] + "áéíóúüñÁÉÍÓÚÜÑ"
VOCABS["norwegian"] = VOCABS["english"] + "æøåÆØÅ"
VOCABS["danish"] = VOCABS["english"] + "æøåÆØÅ"
VOCABS["finnish"] = VOCABS["english"] + "äöÄÖ"
VOCABS["swedish"] = VOCABS["english"] + "åäöÅÄÖ"
VOCABS["vietnamese"] = (
    VOCABS["english"]
    + "áàảạãăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệóòỏõọôốồổộỗơớờởợỡúùủũụưứừửữựiíìỉĩịýỳỷỹỵ"
    + "ÁÀẢẠÃĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÓÒỎÕỌÔỐỒỔỘỖƠỚỜỞỢỠÚÙỦŨỤƯỨỪỬỮỰIÍÌỈĨỊÝỲỶỸỴ"
)
VOCABS["hebrew"] = VOCABS["english"] + "אבגדהוזחטיכלמנסעפצקרשת" + "₪"
VOCABS["hindi"] = VOCABS["hindi_letters"] + VOCABS["hindi_digits"] + VOCABS["hindi_punctuation"]
VOCABS["bangla"] = VOCABS["bangla_letters"] + VOCABS["bangla_digits"]
VOCABS["ukrainian"] = (
    VOCABS["generic_cyrillic_letters"] + VOCABS["digits"] + VOCABS["punctuation"] + VOCABS["currency"] + "ґіїєҐІЇЄ₴"
)
VOCABS["multilingual"] = "".join(
    dict.fromkeys(
        VOCABS["french"]
        + VOCABS["portuguese"]
        + VOCABS["spanish"]
        + VOCABS["german"]
        + VOCABS["czech"]
        + VOCABS["polish"]
        + VOCABS["dutch"]
        + VOCABS["italian"]
        + VOCABS["norwegian"]
        + VOCABS["danish"]
        + VOCABS["finnish"]
        + VOCABS["swedish"]
        + "§"
    )
)

# Added Earlier based on coverage of chars in regular usage
VOCABS['bengali']    = "ঀঁংঃ঄অআইঈউঊঋঌ঍঎এঐ঑঒ওঔকখগঘঙচছজঝঞটঠডঢণতথদধন঩পফবভমযর঱ল঳঴঵শষসহ঺঻়ঽািীুূৃৄ৅৆েৈ৉৊োৌ্ৎ৏৐৑৒৓৔৕৖ৗ৘৙৚৛ড়ঢ়৞য়ৠৡৢৣ৤৥০১২৩৪৫৬৭৮৯ৰৱ৲৳৴৵৶৷৸৹৺৻ৼ৽৾৿"
VOCABS['devanagari'] = "ऀँंःऄअआइईउऊऋऌऍऎएऐऑऒओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळऴवशषसहऺऻ़ऽािीुूृॄॅॆेैॉॊोौ्ॎॏॐ॒॑॓॔ॕॖॗक़ख़ग़ज़ड़ढ़फ़य़ॠॡॢॣ।॥०१२३४५६७८९॰ॱॲॳॴॵॶॷॸॹॺॻॼॽॾॿ" + "꣠꣡꣢꣣꣤꣥꣦꣧꣨꣩꣪꣫꣬꣭꣮꣯꣰꣱ꣲꣳꣴꣵꣶꣷ꣸꣹꣺ꣻ꣼ꣽꣾꣿ"
VOCABS['gujarati']   = "઀ઁંઃ઄અઆઇઈઉઊઋઌઍ઎એઐઑ઒ઓઔકખગઘઙચછજઝઞટઠડઢણતથદધન઩પફબભમયર઱લળ઴વશષસહ઺઻઼ઽાિીુૂૃૄૅ૆ેૈૉ૊ોૌ્૎૏ૐ૑૒૓૔૕૖૗૘૙૚૛૜૝૞૟ૠૡૢૣ૤૥૦૧૨૩૪૫૬૭૮૯૰૱૲૳૴૵૶૷૸ૹૺૻૼ૽૾૿હું"
VOCABS['gurumukhi']  = "਀ਁਂਃ਄ਅਆਇਈਉਊ਋਌਍਎ਏਐ਑਒ਓਔਕਖਗਘਙਚਛਜਝਞਟਠਡਢਣਤਥਦਧਨ਩ਪਫਬਭਮਯਰ਱ਲਲ਼਴ਵਸ਼਷ਸਹ਺਻਼਽ਾਿੀੁੂ੃੄੅੆ੇੈ੉੊ੋੌ੍੎੏੐ੑ੒੓੔੕੖੗੘ਖ਼ਗ਼ਜ਼ੜ੝ਫ਼੟੠੡੢੣੤੥੦੧੨੩੪੫੬੭੮੯ੰੱੲੳੴੵ੶੷੸੹੺੻੼੽੾੿"
VOCABS['kannada']    = "ಀಁಂಃ಄ಅಆಇಈಉಊಋಌ಍ಎಏಐ಑ಒಓಔಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನ಩ಪಫಬಭಮಯರಱಲಳ಴ವಶಷಸಹ಺಻಼ಽಾಿೀುೂೃೄ೅ೆೇೈ೉ೊೋೌ್೎೏೐೑೒೓೔ೕೖ೗೘೙೚೛೜ೝೞ೟ೠೡೢೣ೤೥೦೧೨೩೪೫೬೭೮೯೰ೱೲೳ೴೵೶೷೸೹೺೻೼೽೾೿"
VOCABS['malayalam']  = "ഀഁംഃഄഅആഇഈഉഊഋഌ഍എഏഐ഑ഒഓഔകഖഗഘങചഛജഝഞടഠഡഢണതഥദധനഩപഫബഭമയരറലളഴവശഷസഹഺ഻഼ഽാിീുൂൃൄ൅െേൈ൉ൊോൌ്ൎ൏൐൑൒൓ൔൕൖൗ൘൙൚൛൜൝൞ൟൠൡൢൣ൤൥൦൧൨൩൪൫൬൭൮൯൰൱൲൳൴൵൶൷൸൹ൺൻർൽൾൿ"
VOCABS['odia']       = "଀ଁଂଃ଄ଅଆଇଈଉଊଋଌ଍଎ଏଐ଑଒ଓଔକଖଗଘଙଚଛଜଝଞଟଠଡଢଣତଥଦଧନ଩ପଫବଭମଯର଱ଲଳ଴ଵଶଷସହ଺଻଼ଽାିୀୁୂୃୄ୅୆େୈ୉୊ୋୌ୍୎୏୐୑୒୓୔୕ୖୗ୘୙୚୛ଡ଼ଢ଼୞ୟୠୡୢୣ୤୥୦୧୨୩୪୫୬୭୮୯୰ୱ୲୳୴୵୶୷୸୹୺୻୼୽୾୿"
VOCABS['tamil']      = "஀஁ஂஃ஄அஆஇஈஉஊ஋஌஍எஏஐ஑ஒஓஔக஖஗஘ஙச஛ஜ஝ஞட஠஡஢ணத஥஦஧நனப஫஬஭மயரறலளழவஶஷஸஹ஺஻஼஽ாிீுூ௃௄௅ெேை௉ொோௌ்௎௏ௐ௑௒௓௔௕௖ௗ௘௙௚௛௜௝௞௟௠௡௢௣௤௥௦௧௨௩௪௫௬௭௮௯௰௱௲௳௴௵௶௷௸௹௺௻௼௽௾௿"
VOCABS['telugu']     = "ఀఁంఃఄఅఆఇఈఉఊఋఌ఍ఎఏఐ఑ఒఓఔకఖగఘఙచఛజఝఞటఠడఢణతథదధన఩పఫబభమయరఱలళఴవశషసహ఺఻఼ఽాిీుూృౄ౅ెేై౉ొోౌ్౎౏౐౑౒౓౔ౕౖ౗ౘౙౚ౛౜ౝ౞౟ౠౡౢౣ౤౥౦౧౨౩౪౫౬౭౮౯౰౱౲౳౴౵౶౷౸౹౺౻౼౽౾౿"


# Vocab from Akshara Dataset
VOCABS['akshara_assamese']   = " !\"'()*,-.:;?ABDIKMWabcdefghilmnoprstuvwy।॥ঁংঃঅআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ়ািীুূৃেৈোৌ্ৎড়ঢ়য়০১২৩৪৫৬৭৮৯ৰৱ৷‌"
VOCABS['akshara_bengali']    = " !\"%'()*,-./012345789:;?@ABCDEFGHKLMNPQRSTVW[]abcdefghijklmnoprstuvwxy।ঁংঃঅআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ়ািীুূৃেৈোৌ্ৎড়ঢ়য়০১২৩৪৫৬৭৮৯৷৺‌–—‘’“”"
VOCABS['akshara_gujarati']   = " !\"'(),-./0123456789:;?B]`loy|ंःइकखगजटणतदधनपबमयरलवशषसहािीुूृेैो्।॥ઁંઃઅઆઇઈઉઊઋઍએઐઑઓઔકખગઘઙચછજઝઞટઠડઢણતથદધનપફબભમયરલળવશષસહઽાિીુૂૃૅેૈૉોૌ્ૐૠ૦૧૨૩૪૫૬૭૮૯–“”①★✶﻿"
VOCABS['akshara_gurumukhi']  = " !\"#&'()*,-./0123456789:;?ABCDEFGHIJKLMNOPRSTVWYabcdefghijklmnoprstuvwxy।॥ਂਅਆਇਈਉਊਏਐਓਔਕਖਗਘਙਚਛਜਝਞਟਠਡਢਣਤਥਦਧਨਪਫਬਭਮਯਰਲਲ਼ਵਸ਼ਸਹ਼ਾਿੀੁੂੇੈੋੌ੍ਖ਼ਗ਼ਜ਼ੜਫ਼੦੧੨੩੪੫੬੭੮੯ੰੱੲੳ†"
VOCABS['akshara_hindi']      = " !\"'()*,-./0123456789:;?Sox|ँंःअआइईउऊऋएऐऑओऔकखगघचछजझञटठडढणतथदधनपफबभमयरऱलवशषसह़ऽािीुूृेैॉोौ्क़ख़ग़ज़ड़ढ़फ़।०१२३४५६७८९‌–‘’“”…"
VOCABS['akshara_kannada']    = " !\"%&'()*+,-./0123456789:;=?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]_abcdefghijklmnopqrstuvwxyz|ಂಃಅಆಇಈಉಊಋಎಏಐಒಓಔಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಱಲಳವಶಷಸಹ಼ಽಾಿೀುೂೃೄೆೇೈೊೋೌ್ೞ೦೧೨೩೪೫೬೭೮೯‌‍–‘’“”…→"
VOCABS['akshara_malayalam']  = " !\"'()*,-./0123456789:;=>?ABCDEFGHIJKLMNOPRSTUVWZ[]_abcdefghijklmnopqrstuvwxyzംഃഅആഇഈഉഊഋഎഏഐഒഓഔകഖഗഘങചഛജഝഞടഠഡഢണതഥദധനപഫബഭമയരറലളഴവശഷസഹാിീുൂൃെേൈൊോൌ്ൗൻർൽൾ‌‍‘’“”…"
VOCABS['akshara_manipuri']   = " !&'()+,-./123456789:;=?ABCDEFGHIKLMNOPQRSTUWY\\abcdefghijklmnoprstuvwxyz।ঁংঃঅআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডণতথদধনপফবভমযরলশষসহ়ািীুূৃেৈোৌ্ৎড়য়০১২৩৪৫৬৭৮৯ৱ‌‍‘’“”"
VOCABS['akshara_marathi']    = " !\"$'()*,-./012345789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyzँंःअआइईउऊऋएऐऑओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरऱलळवशषसह़ऽािीुूृॅेैॉोौ्ॐय़।॥०१२३४५६७८९‌‍–‘’“”"
VOCABS['akshara_odia']       = " !\"&'()*,-./0123456789:;?ABCDEFGHIJKLMNPRSTUVWXY_abcdefghijklmnopqrstuvwxyz|।॥ଁଂଃଅଆଇଈଉଊଋଏଐଓଔକଖଗଘଙଚଛଜଝଞଟଠଡଢଣତଥଦଧନପଫବଭମଯରଲଳଵଶଷସହ଼ାିୀୁୂୃେୈୋୌ୍ଡ଼ଢ଼ୟ୦୧୨୩୪୫୬୭୮୯ୱ‌‘’“”"
VOCABS['akshara_tamil']      = " !\"#&'()*+,-./0123456789:;>?@ABCDEFGHIJKLMNOPQRSTUVWXY[]`abcdefghijklmnopqrstuvwxy{}­¹½ïஂஃஅஆஇஈஉஊஎஏஐஒஓஔகஙசஜஞடணதநனபமயரறலளழவஷஸஹாிீுூெேைொோௌ்௨௩௪௫௮௰௳௴–—‘’“”•…"
VOCABS['akshara_telugu']     = " !\"%'()*+,-.0123456789:;=?ACDEGHIKLMNOPQRSTUWYZ[]_abcdefghijklmnoprstuvwxy|ఁంఃఅఆఇఈఉఊఋఎఏఐఒఓఔకఖగఘఙచఛజఝఞటఠడఢణతథదధనపఫబభమయరఱలళవశషసహాిీుూృౄెేైొోౌ్౯"
VOCABS['akshara_urdu']       = " !\"$%'()+,-./0123456789:<>@ABCDEFGHIJKLMNOPQRSTUVWYZ\\_abcdefghijklmnopqrstuvwxyz£©°¶½؁؂،ؐؒؓؔ؟ءآؤئابتثجحخدذرزسشصضطظعغفقلمنوًَُِّٖٓٔ٥٨٭ٰٹپچڈڑژکگںھہیے۔۰۱۲۳۴۵۶۷۸۹ਗਦਨਯਲਸਹਾਿੇੋ‎—‘’“”…ﷺﻩ"
VOCABS['akshara_misc_chars'] = " ­¹½ï।॥‌‍–—‘’“”†•…→①★✶°"

# Vocab from IIIT_INDIC_HW_WORDS Dataset
VOCABS['ihtr_bengali']   = "-।ঁংঃঅআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ়ঽািীুূৃৄেৈোৌ্ৎৗড়ঢ়য়ৠৢৣ০১২৩৪৫৬৭৮৯ৰৱ৲৳৴৵৶৷৹৺৻"
VOCABS['ihtr_gujarati']  = "-ઁંઃઅઆઇઈઉઊઋઍએઐઑઓઔકખગઘઙચછજઝઞટઠડઢણતથદધનપફબભમયરલળવશષસહ઼ઽાિીુૂૃૄૅેૈૉોૌ્ૐ૦૧૨૩૪૫૬૭૮૯૱"
VOCABS['ihtr_gurumukhi'] = "-।ਁਂਃਅਆਇਈਉਊਏਐਓਔਕਖਗਘਙਚਛਜਝਞਟਠਡਢਣਤਥਦਧਨਪਫਬਭਮਯਰਲਲ਼ਵਸ਼ਸਹ਼ਾਿੀੁੂੇੈੋੌ੍ੑਖ਼ਗ਼ਜ਼ੜਫ਼੦੧੨੩੪੫੬੭੮੯ੰੱੲੳੴੵ"
VOCABS['ihtr_hindi']     = "-ँंःअआइईउऊऋऌऍऎएऐऑऒओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळऴवशषसह़ऽािीुूृॄॅेैॉॊोौ्ॐ॒॑॓॔क़ख़ग़ज़ड़ढ़फ़य़ॠॢ।॥०१२३४५६७८९॰ॱॲॻॼॽॾ"
VOCABS['ihtr_kannada']   = "-ಂಃಅಆಇಈಉಊಋಌಎಏಐಒಓಔಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಱಲಳವಶಷಸಹ಼ಾಿೀುೂೃೄೆೇೈೊೋೌ್ೕೖೞೠೢೣ೦೧೨೩೪೫೬೭೮೯"
VOCABS['ihtr_malayalam'] = "-ംഃഅആഇഈഉഊഋഌഎഏഐഒഓഔകഖഗഘങചഛജഝഞടഠഡഢണതഥദധനപഫബഭമയരറലളഴവശഷസഹാിീുൂൃൄെേൈൊോൌ്ൗൠൡൢൣ൦൧൨൩൪൫൬൭൮൯൰൱൲൳൴൵൹ൺൻർൽൾൿ"
VOCABS['ihtr_odia']      = "-।ଁଂଃଅଆଇଈଉଊଋଏଐଓଔକଖଗଘଙଚଛଜଝଞଟଠଡଢଣତଥଦଧନପଫବଭମଯରଲଳଵଶଷସହ଼ାିୀୁୂୃୄେୈୋୌ୍ୖୗଡ଼ଢ଼ୟୠୡୢୣ୦୧୨୩୪୫୬୭୮୯୰ୱ"
VOCABS['ihtr_tamil']     = "-ஃஅஆஇஈஉஊஎஏஐஒஓஔகஙசஜஞடணதநனபமயரறலளழவஶஷஸஹாிீுூெேைொோௌ்ௐௗ௦௧௨௩௪௫௬௭௮௯௰௱௲௳௴௵௶௷௸௹௺"
VOCABS['ihtr_telugu']    = "-ఁంఃఅఆఇఈఉఊఋఎఏఐఒఓఔకఖగఘఙచఛజఝఞటఠడఢణతథదధనపఫబభమయరఱలళవశషసహాిీుూృౄెేైొోౌ్ౘౙౠౢ౦౧౨౩౪౫౬౭౮౯౸౹౺౻౼౽౾"
VOCABS['ihtr_urdu']      = "an،ؑؒؓ؛؟ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىيًٍَُِّْٰٖٗٱٹټپچڈڑژکگںھہۂۃیےۓ۔‌‍‎‏—‘ﷺﺅ"

# Vocab from IIIT_Indic_HW_UC Dataset
VOCABS['uc_assamese'] = ""
VOCABS['uc_bengali'] = ""
VOCABS['uc_gujarati'] = ""
VOCABS['uc_gurumukhi'] = ""
VOCABS["uc_hindi"] = ""
VOCABS['uc_kannada'] = ""
VOCABS['uc_malayalam'] = ""
VOCABS['uc_manipuri'] = ""
VOCABS['uc_marathi'] = ""
VOCABS['uc_odia'] = ""
VOCABS['uc_tamil'] = "!\"#&')*+,-./0123456789:=?ABDEGHIKLMNOPQTUZ^_`acegijlmnoqrsuxy|~·ÄÉàäஃஅஆஇஈஉஊஎஏஐஒஓஔகஙசஜஞடணதநனபமயரறலளழவஷஸஹாிீுூெேைொோௌ்ௗ௧௨௩௪௫௭௮௱௹‘’“”•−",
VOCABS['uc_telugu'] = ""
VOCABS['uc_urdu'] = ""


# Vocab from Unicode Ranges through scripts
VOCABS['unicode_assamese_bengali']      = "ঀঁংঃ঄অআইঈউঊঋঌ঍঎এঐ঑঒ওঔকখগঘঙচছজঝঞটঠডঢণতথদধন঩পফবভমযর঱ল঳঴঵শষসহ঺঻়ঽািীুূৃৄ৅৆েৈ৉৊োৌ্ৎ৏৐৑৒৓৔৕৖ৗ৘৙৚৛ড়ঢ়৞য়ৠৡৢৣ৤৥০১২৩৪৫৬৭৮৯ৰৱ৲৳৴৵৶৷৸৹৺৻ৼ৽৾৿"

VOCABS['unicode_devanagari']            = "ऀँंःऄअआइईउऊऋऌऍऎएऐऑऒओऔकखगघङचछजझञटठडढणतथदधनऩपफबभमयरऱलळऴवशषसहऺऻ़ऽािीुूृॄॅॆेैॉॊोौ्ॎॏॐ॒॑॓॔ॕॖॗक़ख़ग़ज़ड़ढ़फ़य़ॠॡॢॣ।॥०१२३४५६७८९॰ॱॲॳॴॵॶॷॸॹॺॻॼॽॾॿ"
VOCABS['unicode_devanagari_extended']   = "꣠꣡꣢꣣꣤꣥꣦꣧꣨꣩꣪꣫꣬꣭꣮꣯꣰꣱ꣲꣳꣴꣵꣶꣷ꣸꣹꣺ꣻ꣼ꣽꣾꣿ"
VOCABS["unicode_devanagari_extended_A"] ="𑬀𑬁𑬂𑬃𑬄𑬅𑬆𑬇𑬈𑬉𑬊𑬋𑬌𑬍𑬎𑬏𑬐𑬑𑬒𑬓𑬔𑬕𑬖𑬗𑬘𑬙𑬚𑬛𑬜𑬝𑬞𑬟𑬠𑬡𑬢𑬣𑬤𑬥𑬦𑬧𑬨𑬩𑬪𑬫𑬬𑬭𑬮𑬯𑬰𑬱𑬲𑬳𑬴𑬵𑬶𑬷𑬸𑬹𑬺𑬻𑬼𑬽𑬾𑬿𑭀𑭁𑭂𑭃𑭄𑭅𑭆𑭇𑭈𑭉𑭊𑭋𑭌𑭍𑭎𑭏𑭐𑭑𑭒𑭓𑭔𑭕𑭖𑭗𑭘𑭙𑭚𑭛𑭜𑭝𑭞𑭟",

VOCABS['unicode_tamil']                 = "஀஁ஂஃ஄அஆஇஈஉஊ஋஌஍எஏஐ஑ஒஓஔக஖஗஘ஙச஛ஜ஝ஞட஠஡஢ணத஥஦஧நனப஫஬஭மயரறலளழவஶஷஸஹ஺஻஼஽ாிீுூ௃௄௅ெேை௉ொோௌ்௎௏ௐ௑௒௓௔௕௖ௗ௘௙௚௛௜௝௞௟௠௡௢௣௤௥௦௧௨௩௪௫௬௭௮௯௰௱௲௳௴௵௶௷௸௹௺௻௼௽௾௿"
VOCABS['unicode_tamil_supplement']      = "𑿀𑿁𑿂𑿃𑿄𑿅𑿆𑿇𑿈𑿉𑿊𑿋𑿌𑿍𑿎𑿏𑿐𑿑𑿒𑿓𑿔𑿕𑿖𑿗𑿘𑿙𑿚𑿛𑿜𑿝𑿞𑿟𑿠𑿡𑿢𑿣𑿤𑿥𑿦𑿧𑿨𑿩𑿪𑿫𑿬𑿭𑿮𑿯𑿰𑿱𑿲𑿳𑿴𑿵𑿶𑿷𑿸𑿹𑿺𑿻𑿼𑿽𑿾𑿿"

VOCABS['unicode_gujarati']              = "઀ઁંઃ઄અઆઇઈઉઊઋઌઍ઎એઐઑ઒ઓઔકખગઘઙચછજઝઞટઠડઢણતથદધન઩પફબભમયર઱લળ઴વશષસહ઺઻઼ઽાિીુૂૃૄૅ૆ેૈૉ૊ોૌ્૎૏ૐ૑૒૓૔૕૖૗૘૙૚૛૜૝૞૟ૠૡૢૣ૤૥૦૧૨૩૪૫૬૭૮૯૰૱૲૳૴૵૶૷૸ૹૺૻૼ૽૾૿"
VOCABS['unicode_gurumukhi']             = "਀ਁਂਃ਄ਅਆਇਈਉਊ਋਌਍਎ਏਐ਑਒ਓਔਕਖਗਘਙਚਛਜਝਞਟਠਡਢਣਤਥਦਧਨ਩ਪਫਬਭਮਯਰ਱ਲਲ਼਴ਵਸ਼਷ਸਹ਺਻਼਽ਾਿੀੁੂ੃੄੅੆ੇੈ੉੊ੋੌ੍੎੏੐ੑ੒੓੔੕੖੗੘ਖ਼ਗ਼ਜ਼ੜ੝ਫ਼੟੠੡੢੣੤੥੦੧੨੩੪੫੬੭੮੯ੰੱੲੳੴੵ੶੷੸੹੺੻੼੽੾੿" 
VOCABS['unicode_kannada']               = "ಀಁಂಃ಄ಅಆಇಈಉಊಋಌ಍ಎಏಐ಑ಒಓಔಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನ಩ಪಫಬಭಮಯರಱಲಳ಴ವಶಷಸಹ಺಻಼ಽಾಿೀುೂೃೄ೅ೆೇೈ೉ೊೋೌ್೎೏೐೑೒೓೔ೕೖ೗೘೙೚೛೜ೝೞ೟ೠೡೢೣ೤೥೦೧೨೩೪೫೬೭೮೯೰ೱೲೳ೴೵೶೷೸೹೺೻೼೽೾೿"
VOCABS['unicode_malayalam']             = "ഀഁംഃഄഅആഇഈഉഊഋഌ഍എഏഐ഑ഒഓഔകഖഗഘങചഛജഝഞടഠഡഢണതഥദധനഩപഫബഭമയരറലളഴവശഷസഹഺ഻഼ഽാിീുൂൃൄ൅െേൈ൉ൊോൌ്ൎ൏൐൑൒൓ൔൕൖൗ൘൙൚൛൜൝൞ൟൠൡൢൣ൤൥൦൧൨൩൪൫൬൭൮൯൰൱൲൳൴൵൶൷൸൹ൺൻർൽൾൿ"
VOCABS['unicode_odia']                  = "଀ଁଂଃ଄ଅଆଇଈଉଊଋଌ଍଎ଏଐ଑଒ଓଔକଖଗଘଙଚଛଜଝଞଟଠଡଢଣତଥଦଧନ଩ପଫବଭମଯର଱ଲଳ଴ଵଶଷସହ଺଻଼ଽାିୀୁୂୃୄ୅୆େୈ୉୊ୋୌ୍୎୏୐୑୒୓୔୕ୖୗ୘୙୚୛ଡ଼ଢ଼୞ୟୠୡୢୣ୤୥୦୧୨୩୪୫୬୭୮୯୰ୱ୲୳୴୵୶୷୸୹୺୻୼୽୾୿"
VOCABS['unicode_telugu']                = "ఀఁంఃఄఅఆఇఈఉఊఋఌ఍ఎఏఐ఑ఒఓఔకఖగఘఙచఛజఝఞటఠడఢణతథదధన఩పఫబభమయరఱలళఴవశషసహ఺఻఼ఽాిీుూృౄ౅ెేై౉ొోౌ్౎౏౐౑౒౓౔ౕౖ౗ౘౙౚ౛౜ౝ౞౟ౠౡౢౣ౤౥౦౧౨౩౪౫౬౭౮౯౰౱౲౳౴౵౶౷౸౹౺౻౼౽౾౿"





VOCABS['unicode_arabic']                = "؀؁؂؃؄؅؆؇؈؉؊؋،؍؎؏ؘؙؚؐؑؒؓؔؕؖؗ؛؜؝؞؟ؠءآأؤإئابةتثجحخدذرزسشصضطظعغػؼؽؾؿـفقكلمنهوىيًٌٍَُِّْٕٖٜٟٓٔٗ٘ٙٚٛٝٞ٠١٢٣٤٥٦٧٨٩٪٫٬٭ٮٯٰٱٲٳٴٵٶٷٸٹٺٻټٽپٿڀځڂڃڄڅچڇڈډڊڋڌڍڎڏڐڑڒړڔڕږڗژڙښڛڜڝڞڟڠڡڢڣڤڥڦڧڨکڪګڬڭڮگڰڱڲڳڴڵڶڷڸڹںڻڼڽھڿۀہۂۃۄۅۆۇۈۉۊۋیۍێۏېۑےۓ۔ەۖۗۘۙۚۛۜ۝۞ۣ۟۠ۡۢۤۥۦۧۨ۩۪ۭ۫۬ۮۯ۰۱۲۳۴۵۶۷۸۹ۺۻۼ۽۾ۿ"


# Low Resource Scripts of India
VOCABS["unicode_ahom"]                    = "𑜀𑜁𑜂𑜃𑜄𑜅𑜆𑜇𑜈𑜉𑜊𑜋𑜌𑜍𑜎𑜏𑜐𑜑𑜒𑜓𑜔𑜕𑜖𑜗𑜘𑜙𑜚𑜛𑜜𑜝𑜞𑜟𑜠𑜡𑜢𑜣𑜤𑜥𑜦𑜧𑜨𑜩𑜪𑜫𑜬𑜭𑜮𑜯𑜰𑜱𑜲𑜳𑜴𑜵𑜶𑜷𑜸𑜹𑜺𑜻𑜼𑜽𑜾𑜿𑝀𑝁𑝂𑝃𑝄𑝅𑝆𑝇𑝈𑝉𑝊𑝋𑝌𑝍𑝎𑝏"
VOCABS["unicode_bhaiksuki"]               = "𑰀𑰁𑰂𑰃𑰄𑰅𑰆𑰇𑰈𑰉𑰊𑰋𑰌𑰍𑰎𑰏𑰐𑰑𑰒𑰓𑰔𑰕𑰖𑰗𑰘𑰙𑰚𑰛𑰜𑰝𑰞𑰟𑰠𑰡𑰢𑰣𑰤𑰥𑰦𑰧𑰨𑰩𑰪𑰫𑰬𑰭𑰮𑰯𑰰𑰱𑰲𑰳𑰴𑰵𑰶𑰷𑰸𑰹𑰺𑰻𑰼𑰽𑰾𑰿𑱀𑱁𑱂𑱃𑱄𑱅𑱆𑱇𑱈𑱉𑱊𑱋𑱌𑱍𑱎𑱏𑱐𑱑𑱒𑱓𑱔𑱕𑱖𑱗𑱘𑱙𑱚𑱛𑱜𑱝𑱞𑱟𑱠𑱡𑱢𑱣𑱤𑱥𑱦𑱧𑱨𑱩𑱪𑱫𑱬𑱭𑱮𑱯"
VOCABS["unicode_brahmi"]                  = "𑀀𑀁𑀂𑀃𑀄𑀅𑀆𑀇𑀈𑀉𑀊𑀋𑀌𑀍𑀎𑀏𑀐𑀑𑀒𑀓𑀔𑀕𑀖𑀗𑀘𑀙𑀚𑀛𑀜𑀝𑀞𑀟𑀠𑀡𑀢𑀣𑀤𑀥𑀦𑀧𑀨𑀩𑀪𑀫𑀬𑀭𑀮𑀯𑀰𑀱𑀲𑀳𑀴𑀵𑀶𑀷𑀸𑀹𑀺𑀻𑀼𑀽𑀾𑀿𑁀𑁁𑁂𑁃𑁄𑁅𑁆𑁇𑁈𑁉𑁊𑁋𑁌𑁍𑁎𑁏𑁐𑁑𑁒𑁓𑁔𑁕𑁖𑁗𑁘𑁙𑁚𑁛𑁜𑁝𑁞𑁟𑁠𑁡𑁢𑁣𑁤𑁥𑁦𑁧𑁨𑁩𑁪𑁫𑁬𑁭𑁮𑁯𑁰𑁱𑁲𑁳𑁴𑁵𑁶𑁷𑁸𑁹𑁺𑁻𑁼𑁽𑁾𑁿"
VOCABS["unicode_chakma"]                  = "𑄀𑄁𑄂𑄃𑄄𑄅𑄆𑄇𑄈𑄉𑄊𑄋𑄌𑄍𑄎𑄏𑄐𑄑𑄒𑄓𑄔𑄕𑄖𑄗𑄘𑄙𑄚𑄛𑄜𑄝𑄞𑄟𑄠𑄡𑄢𑄣𑄤𑄥𑄦𑄧𑄨𑄩𑄪𑄫𑄬𑄭𑄮𑄯𑄰𑄱𑄲𑄳𑄴𑄵𑄶𑄷𑄸𑄹𑄺𑄻𑄼𑄽𑄾𑄿𑅀𑅁𑅂𑅃𑅄𑅅𑅆𑅇𑅈𑅉𑅊𑅋𑅌𑅍𑅎𑅏"
VOCABS["unicode_dogra"]                   = "𑠀𑠁𑠂𑠃𑠄𑠅𑠆𑠇𑠈𑠉𑠊𑠋𑠌𑠍𑠎𑠏𑠐𑠑𑠒𑠓𑠔𑠕𑠖𑠗𑠘𑠙𑠚𑠛𑠜𑠝𑠞𑠟𑠠𑠡𑠢𑠣𑠤𑠥𑠦𑠧𑠨𑠩𑠪𑠫𑠬𑠭𑠮𑠯𑠰𑠱𑠲𑠳𑠴𑠵𑠶𑠷𑠸𑠺𑠹𑠻𑠼𑠽𑠾𑠿𑡀𑡁𑡂𑡃𑡄𑡅𑡆𑡇𑡈𑡉𑡊𑡋𑡌𑡍𑡎𑡏"
VOCABS["unicode_grantha"]                 = "𑌀𑌁𑌂𑌃𑌄𑌅𑌆𑌇𑌈𑌉𑌊𑌋𑌌𑌍𑌎𑌏𑌐𑌑𑌒𑌓𑌔𑌕𑌖𑌗𑌘𑌙𑌚𑌛𑌜𑌝𑌞𑌟𑌠𑌡𑌢𑌣𑌤𑌥𑌦𑌧𑌨𑌩𑌪𑌫𑌬𑌭𑌮𑌯𑌰𑌱𑌲𑌳𑌴𑌵𑌶𑌷𑌸𑌹𑌺𑌻𑌼𑌽𑌾𑌿𑍀𑍁𑍂𑍃𑍄𑍅𑍆𑍇𑍈𑍉𑍊𑍋𑍌𑍍𑍎𑍏𑍐𑍑𑍒𑍓𑍔𑍕𑍖𑍗𑍘𑍙𑍚𑍛𑍜𑍝𑍞𑍟𑍠𑍡𑍢𑍣𑍤𑍥𑍦𑍧𑍨𑍩𑍪𑍫𑍬𑍭𑍮𑍯𑍰𑍱𑍲𑍳𑍴𑍵𑍶𑍷𑍸𑍹𑍺𑍻𑍼𑍽𑍾𑍿"
VOCABS["unicode_gunjala_gondi"]           = "𑵠𑵡𑵢𑵣𑵤𑵥𑵦𑵧𑵨𑵩𑵪𑵫𑵬𑵭𑵮𑵯𑵰𑵱𑵲𑵳𑵴𑵵𑵶𑵷𑵸𑵹𑵺𑵻𑵼𑵽𑵾𑵿𑶀𑶁𑶂𑶃𑶄𑶅𑶆𑶇𑶈𑶉𑶊𑶋𑶌𑶍𑶎𑶏𑶐𑶑𑶒𑶓𑶔𑶕𑶖𑶗𑶘𑶙𑶚𑶛𑶜𑶝𑶞𑶟𑶠𑶡𑶢𑶣𑶤𑶥𑶦𑶧𑶨𑶩𑶪𑶫𑶬𑶭𑶮𑶯"
VOCABS["unicode_kaithi"]                  = "𑂀𑂁𑂂𑂃𑂄𑂅𑂆𑂇𑂈𑂉𑂊𑂋𑂌𑂍𑂎𑂏𑂐𑂑𑂒𑂓𑂔𑂕𑂖𑂗𑂘𑂙𑂚𑂛𑂜𑂝𑂞𑂟𑂠𑂡𑂢𑂣𑂤𑂥𑂦𑂧𑂨𑂩𑂪𑂫𑂬𑂭𑂮𑂯𑂰𑂱𑂲𑂳𑂴𑂵𑂶𑂷𑂸𑂺𑂹𑂻𑂼𑂽𑂾𑂿𑃀𑃁𑃂𑃃𑃄𑃅𑃆𑃇𑃈𑃉𑃊𑃋𑃌𑃍𑃎𑃏"
VOCABS["unicode_khojki"]                  = "𑈀𑈁𑈂𑈃𑈄𑈅𑈆𑈇𑈈𑈉𑈊𑈋𑈌𑈍𑈎𑈏𑈐𑈑𑈒𑈓𑈔𑈕𑈖𑈗𑈘𑈙𑈚𑈛𑈜𑈝𑈞𑈟𑈠𑈡𑈢𑈣𑈤𑈥𑈦𑈧𑈨𑈩𑈪𑈫𑈬𑈭𑈮𑈯𑈰𑈱𑈲𑈳𑈴𑈶𑈵𑈷𑈸𑈹𑈺𑈻𑈼𑈽𑈾𑈿𑉀𑉁𑉂𑉃𑉄𑉅𑉆𑉇𑉈𑉉𑉊𑉋𑉌𑉍𑉎𑉏"
VOCABS["unicode_lepcha"]                  = "ᰀᰁᰂᰃᰄᰅᰆᰇᰈᰉᰊᰋᰌᰍᰎᰏᰐᰑᰒᰓᰔᰕᰖᰗᰘᰙᰚᰛᰜᰝᰞᰟᰠᰡᰢᰣᰤᰥᰦᰧᰨᰩᰪᰫᰬᰭᰮᰯᰰᰱᰲᰳᰴᰵᰶ᰷᰸᰹᰺᰻᰼᰽᰾᰿᱀᱁᱂᱃᱄᱅᱆᱇᱈᱉᱊᱋᱌ᱍᱎᱏ"
VOCABS["unicode_limbu"]                   = "ᤀᤁᤂᤃᤄᤅᤆᤇᤈᤉᤊᤋᤌᤍᤎᤏᤐᤑᤒᤓᤔᤕᤖᤗᤘᤙᤚᤛᤜᤝᤞ᤟ᤠᤡᤢᤣᤤᤥᤦᤧᤨᤩᤪᤫ᤬᤭᤮᤯ᤰᤱᤲᤳᤴᤵᤶᤷᤸ᤻᤹᤺᤼᤽᤾᤿᥀᥁᥂᥃᥄᥅᥆᥇᥈᥉᥊᥋᥌᥍᥎᥏"
VOCABS["unicode_mahajani"]                = "𑅐𑅑𑅒𑅓𑅔𑅕𑅖𑅗𑅘𑅙𑅚𑅛𑅜𑅝𑅞𑅟𑅠𑅡𑅢𑅣𑅤𑅥𑅦𑅧𑅨𑅩𑅪𑅫𑅬𑅭𑅮𑅯𑅰𑅱𑅲𑅳𑅴𑅵𑅶𑅷𑅸𑅹𑅺𑅻𑅼𑅽𑅾𑅿"
VOCABS["unicode_masaram_gondi"]           = "𑴀𑴁𑴂𑴃𑴄𑴅𑴆𑴇𑴈𑴉𑴊𑴋𑴌𑴍𑴎𑴏𑴐𑴑𑴒𑴓𑴔𑴕𑴖𑴗𑴘𑴙𑴚𑴛𑴜𑴝𑴞𑴟𑴠𑴡𑴢𑴣𑴤𑴥𑴦𑴧𑴨𑴩𑴪𑴫𑴬𑴭𑴮𑴯𑴰𑴱𑴲𑴳𑴴𑴵𑴶𑴷𑴸𑴹𑴺𑴻𑴼𑴽𑴾𑴿𑵀𑵁𑵂𑵃𑵄𑵅𑵆𑵇𑵈𑵉𑵊𑵋𑵌𑵍𑵎𑵏𑵐𑵑𑵒𑵓𑵔𑵕𑵖𑵗𑵘𑵙𑵚𑵛𑵜𑵝𑵞𑵟"
VOCABS["unicode_meetei_mayek"]            = "ꯀꯁꯂꯃꯄꯅꯆꯇꯈꯉꯊꯋꯌꯍꯎꯏꯐꯑꯒꯓꯔꯕꯖꯗꯘꯙꯚꯛꯜꯝꯞꯟꯠꯡꯢꯣꯤꯥꯦꯧꯨꯩꯪ꯫꯬꯭꯮꯯꯰꯱꯲꯳꯴꯵꯶꯷꯸꯹꯺꯻꯼꯽꯾꯿"
VOCABS["unicode_meetei_mayek_extensions"] = "ꫠꫡꫢꫣꫤꫥꫦꫧꫨꫩꫪꫫꫬꫭꫮꫯ꫰꫱ꫲꫳꫴꫵ꫶꫷꫸꫹꫺꫻꫼꫽꫾꫿"
VOCABS["unicode_modi"]                    = "𑘀𑘁𑘂𑘃𑘄𑘅𑘆𑘇𑘈𑘉𑘊𑘋𑘌𑘍𑘎𑘏𑘐𑘑𑘒𑘓𑘔𑘕𑘖𑘗𑘘𑘙𑘚𑘛𑘜𑘝𑘞𑘟𑘠𑘡𑘢𑘣𑘤𑘥𑘦𑘧𑘨𑘩𑘪𑘫𑘬𑘭𑘮𑘯𑘰𑘱𑘲𑘳𑘴𑘵𑘶𑘷𑘸𑘹𑘺𑘻𑘼𑘽𑘾𑘿𑙀𑙁𑙂𑙃𑙄𑙅𑙆𑙇𑙈𑙉𑙊𑙋𑙌𑙍𑙎𑙏𑙐𑙑𑙒𑙓𑙔𑙕𑙖𑙗𑙘𑙙𑙚𑙛𑙜𑙝𑙞𑙟"
VOCABS["unicode_nag_mundari"]             = "𞓐𞓑𞓒𞓓𞓔𞓕𞓖𞓗𞓘𞓙𞓚𞓛𞓜𞓝𞓞𞓟𞓠𞓡𞓢𞓣𞓤𞓥𞓦𞓧𞓨𞓩𞓪𞓫𞓮𞓯𞓬𞓭𞓰𞓱𞓲𞓳𞓴𞓵𞓶𞓷𞓸𞓹𞓺𞓻𞓼𞓽𞓾𞓿"
VOCABS["unicode_nandinagari"]             = "𑦠𑦡𑦢𑦣𑦤𑦥𑦦𑦧𑦨𑦩𑦪𑦫𑦬𑦭𑦮𑦯𑦰𑦱𑦲𑦳𑦴𑦵𑦶𑦷𑦸𑦹𑦺𑦻𑦼𑦽𑦾𑦿𑧀𑧁𑧂𑧃𑧄𑧅𑧆𑧇𑧈𑧉𑧊𑧋𑧌𑧍𑧎𑧏𑧐𑧑𑧒𑧓𑧔𑧕𑧖𑧗𑧘𑧙𑧚𑧛𑧜𑧝𑧞𑧟𑧠𑧡𑧢𑧣𑧤𑧥𑧦𑧧𑧨𑧩𑧪𑧫𑧬𑧭𑧮𑧯𑧰𑧱𑧲𑧳𑧴𑧵𑧶𑧷𑧸𑧹𑧺𑧻𑧼𑧽𑧾𑧿"
VOCABS["unicode_ol_chiki"]                = "᱐᱑᱒᱓᱔᱕᱖᱗᱘᱙ᱚᱛᱜᱝᱞᱟᱠᱡᱢᱣᱤᱥᱦᱧᱨᱩᱪᱫᱬᱭᱮᱯᱰᱱᱲᱳᱴᱵᱶᱷᱸᱹᱺᱻᱼᱽ᱾᱿"
VOCABS["unicode_saurashtra"]              = "ꢀꢁꢂꢃꢄꢅꢆꢇꢈꢉꢊꢋꢌꢍꢎꢏꢐꢑꢒꢓꢔꢕꢖꢗꢘꢙꢚꢛꢜꢝꢞꢟꢠꢡꢢꢣꢤꢥꢦꢧꢨꢩꢪꢫꢬꢭꢮꢯꢰꢱꢲꢳꢴꢵꢶꢷꢸꢹꢺꢻꢼꢽꢾꢿꣀꣁꣂꣃ꣄ꣅ꣆꣇꣈꣉꣊꣋꣌꣍꣎꣏꣐꣑꣒꣓꣔꣕꣖꣗꣘꣙꣚꣛꣜꣝꣞꣟"
VOCABS["unicode_sharada"]                 = "𑆀𑆁𑆂𑆃𑆄𑆅𑆆𑆇𑆈𑆉𑆊𑆋𑆌𑆍𑆎𑆏𑆐𑆑𑆒𑆓𑆔𑆕𑆖𑆗𑆘𑆙𑆚𑆛𑆜𑆝𑆞𑆟𑆠𑆡𑆢𑆣𑆤𑆥𑆦𑆧𑆨𑆩𑆪𑆫𑆬𑆭𑆮𑆯𑆰𑆱𑆲𑆳𑆴𑆵𑆶𑆷𑆸𑆹𑆺𑆻𑆼𑆽𑆾𑆿𑇀𑇁𑇂𑇃𑇄𑇅𑇆𑇇𑇈𑇉𑇊𑇋𑇌𑇍𑇎𑇏𑇐𑇑𑇒𑇓𑇔𑇕𑇖𑇗𑇘𑇙𑇚𑇛𑇜𑇝𑇞𑇟"
VOCABS["unicode_siddham"]                 = "𑖀𑖁𑖂𑖃𑖄𑖅𑖆𑖇𑖈𑖉𑖊𑖋𑖌𑖍𑖎𑖏𑖐𑖑𑖒𑖓𑖔𑖕𑖖𑖗𑖘𑖙𑖚𑖛𑖜𑖝𑖞𑖟𑖠𑖡𑖢𑖣𑖤𑖥𑖦𑖧𑖨𑖩𑖪𑖫𑖬𑖭𑖮𑖯𑖰𑖱𑖲𑖳𑖴𑖵𑖶𑖷𑖸𑖹𑖺𑖻𑖼𑖽𑖾𑗀𑖿𑗁𑗂𑗃𑗄𑗅𑗆𑗇𑗈𑗉𑗊𑗋𑗌𑗍𑗎𑗏𑗐𑗑𑗒𑗓𑗔𑗕𑗖𑗗𑗘𑗙𑗚𑗛𑗜𑗝𑗞𑗟𑗠𑗡𑗢𑗣𑗤𑗥𑗦𑗧𑗨𑗩𑗪𑗫𑗬𑗭𑗮𑗯𑗰𑗱𑗲𑗳𑗴𑗵𑗶𑗷𑗸𑗹𑗺𑗻𑗼𑗽𑗾𑗿"
VOCABS["unicode_sora_sompeng"]            = "𑃐𑃑𑃒𑃓𑃔𑃕𑃖𑃗𑃘𑃙𑃚𑃛𑃜𑃝𑃞𑃟𑃠𑃡𑃢𑃣𑃤𑃥𑃦𑃧𑃨𑃩𑃪𑃫𑃬𑃭𑃮𑃯𑃰𑃱𑃲𑃳𑃴𑃵𑃶𑃷𑃸𑃹𑃺𑃻𑃼𑃽𑃾𑃿"
VOCABS["unicode_syloti_nagri"]            = "ꠀꠁꠂꠃꠄꠅ꠆ꠇꠈꠉꠊꠋꠌꠍꠎꠏꠐꠑꠒꠓꠔꠕꠖꠗꠘꠙꠚꠛꠜꠝꠞꠟꠠꠡꠢꠣꠤꠥꠦꠧ꠨꠩꠪꠫꠬꠭꠮꠯"
VOCABS["unicode_takri"]                   = "𑚀𑚁𑚂𑚃𑚄𑚅𑚆𑚇𑚈𑚉𑚊𑚋𑚌𑚍𑚎𑚏𑚐𑚑𑚒𑚓𑚔𑚕𑚖𑚗𑚘𑚙𑚚𑚛𑚜𑚝𑚞𑚟𑚠𑚡𑚢𑚣𑚤𑚥𑚦𑚧𑚨𑚩𑚪𑚫𑚬𑚭𑚮𑚯𑚰𑚱𑚲𑚳𑚴𑚵𑚷𑚶𑚸𑚹𑚺𑚻𑚼𑚽𑚾𑚿𑛀𑛁𑛂𑛃𑛄𑛅𑛆𑛇𑛈𑛉𑛊𑛋𑛌𑛍𑛎𑛏"
VOCABS["unicode_tirhuta"]                 = "𑒀𑒁𑒂𑒃𑒄𑒅𑒆𑒇𑒈𑒉𑒊𑒋𑒌𑒍𑒎𑒏𑒐𑒑𑒒𑒓𑒔𑒕𑒖𑒗𑒘𑒙𑒚𑒛𑒜𑒝𑒞𑒟𑒠𑒡𑒢𑒣𑒤𑒥𑒦𑒧𑒨𑒩𑒪𑒫𑒬𑒭𑒮𑒯𑒰𑒱𑒲𑒳𑒴𑒵𑒶𑒷𑒸𑒻𑒻𑒼𑒽𑒾𑒿𑓀𑓁𑓃𑓂𑓄𑓅𑓆𑓇𑓈𑓉𑓊𑓋𑓌𑓍𑓎𑓏𑓐𑓑𑓒𑓓𑓔𑓕𑓖𑓗𑓘𑓙𑓚𑓛𑓜𑓝𑓞𑓟"
VOCABS["unicode_toto"]                    = "𞊐𞊑𞊒𞊓𞊔𞊕𞊖𞊗𞊘𞊙𞊚𞊛𞊜𞊝𞊞𞊟𞊠𞊡𞊢𞊣𞊤𞊥𞊦𞊧𞊨𞊩𞊪𞊫𞊬𞊭𞊮𞊯𞊰𞊱𞊲𞊳𞊴𞊵𞊶𞊷𞊸𞊹𞊺𞊻𞊼𞊽𞊾𞊿"
VOCABS["unicode_vedic_extensions"]        = "᳐᳑᳒᳓᳔᳕᳖᳗᳘᳙᳜᳝᳞᳟᳚᳛᳠᳡᳢᳣᳤᳥᳦᳧᳨ᳩᳪᳫᳬ᳭ᳮᳯᳰᳱᳲᳳ᳴ᳵᳶ᳷᳸᳹ᳺ᳻᳼᳽᳾᳿"
VOCABS["unicode_wancho"]                  = "𞋀𞋁𞋂𞋃𞋄𞋅𞋆𞋇𞋈𞋉𞋊𞋋𞋌𞋍𞋎𞋏𞋐𞋑𞋒𞋓𞋔𞋕𞋖𞋗𞋘𞋙𞋚𞋛𞋜𞋝𞋞𞋟𞋠𞋡𞋢𞋣𞋤𞋥𞋦𞋧𞋨𞋩𞋪𞋫𞋬𞋭𞋮𞋯𞋰𞋱𞋲𞋳𞋴𞋵𞋶𞋷𞋸𞋹𞋺𞋻𞋼𞋽𞋾𞋿"
VOCABS["unicode_warang_citi"]             = "𑢠𑢡𑢢𑢣𑢤𑢥𑢦𑢧𑢨𑢩𑢪𑢫𑢬𑢭𑢮𑢯𑢰𑢱𑢲𑢳𑢴𑢵𑢶𑢷𑢸𑢹𑢺𑢻𑢼𑢽𑢾𑢿𑣀𑣁𑣂𑣃𑣄𑣅𑣆𑣇𑣈𑣉𑣊𑣋𑣌𑣍𑣎𑣏𑣐𑣑𑣒𑣓𑣔𑣕𑣖𑣗𑣘𑣙𑣚𑣛𑣜𑣝𑣞𑣟𑣠𑣡𑣢𑣣𑣤𑣥𑣦𑣧𑣨𑣩𑣪𑣫𑣬𑣭𑣮𑣯𑣰𑣱𑣲𑣳𑣴𑣵𑣶𑣷𑣸𑣹𑣺𑣻𑣼𑣽𑣾𑣿",

# VOCAB for official 22 Indian languages
VOCABS['unicode_assamese']  = VOCABS['unicode_assamese_bengali']
VOCABS['unicode_bengali']   = VOCABS['unicode_assamese_bengali']
VOCABS['unicode_bodo']      = VOCABS['unicode_devanagari'] + VOCABS['unicode_devanagari_extended'] 
VOCABS['unicode_dogri']     = VOCABS['unicode_dogra'] + VOCABS['unicode_mahajani'] + VOCABS['unicode_devanagari'] + VOCABS['unicode_devanagari_extended']
VOCABS['unicode_gujarati']  = VOCABS['unicode_gujarati']
VOCABS['unicode_hindi']     = VOCABS['unicode_devanagari'] + VOCABS['unicode_devanagari_extended'] 
VOCABS['unicode_kannada']   = VOCABS['unicode_kannada']
VOCABS['unicode_kashmiri']  = VOCABS['unicode_arabic'] + VOCABS['unicode_devanagari'] + VOCABS['unicode_sharada']
VOCABS['unicode_konkani']   = VOCABS['unicode_devanagari'] + VOCABS['unicode_devanagari_extended'] + VOCABS['unicode_kannada'] + VOCABS['unicode_malayalam'] + VOCABS['unicode_arabic']
VOCABS['unicode_maithili']  = VOCABS['unicode_tirhuta'] + VOCABS['unicode_devanagari'] + VOCABS['unicode_devanagari_extended']
VOCABS['unicode_malayalam'] = VOCABS['unicode_malayalam'] + VOCABS['unicode_grantha']
VOCABS['unicode_manipuri']  = VOCABS['unicode_meetei_mayek'] + VOCABS['unicode_meetei_mayek_extensions'] + VOCABS['unicode_bengali']
VOCABS['unicode_marathi']   = VOCABS['unicode_devanagari'] + VOCABS['unicode_devanagari_extended'] + VOCABS['unicode_modi']
VOCABS['unicode_nepali']    = VOCABS['unicode_devanagari'] + VOCABS['unicode_devanagari_extended']
VOCABS['unicode_odia']      = VOCABS['unicode_odia']
VOCABS['unicode_punjabi']   = VOCABS['unicode_gurumukhi']
VOCABS['unicode_sanskrit']  = VOCABS['unicode_devanagari'] + VOCABS['unicode_devanagari_extended'] + VOCABS['unicode_brahmi'] 
VOCABS['unicode_santali']   = VOCABS['unicode_ol_chiki'] + VOCABS['unicode_assamese_bengali'] + VOCABS['odia']
VOCABS['unicode_sindhi']    = VOCABS['unicode_devanagari'] + VOCABS['unicode_devanagari_extended'] + VOCABS['unicode_arabic']
VOCABS['unicode_tamil']     = VOCABS['unicode_tamil'] + VOCABS['unicode_tamil_supplement']
VOCABS['unicode_telugu']    = VOCABS['unicode_telugu']
VOCABS['unicode_urdu']      = VOCABS['unicode_arabic']


VOCABS['unicode_hindi_historical']   = VOCABS['unicode_hindi'] + VOCABS['unicode_kaithi'] + VOCABS["unicode_mahajani"]
VOCABS['unicode_tamil_historical']   = VOCABS['unicode_tamil'] + VOCABS['unicode_grantha'] + VOCABS['unicode_brahmi']
VOCABS['unicode_konkani_historical'] = VOCABS['unicode_konkani'] + VOCABS['unicode_brahmi'] + VOCABS['unicode_modi']





# Final Vocab list
# VOCAB for official 22 Indian languages
VOCABS['all_assamese']  = VOCABS['english'] + VOCABS['unicode_assamese'] + VOCABS['bengali'] + VOCABS['ihtr_bengali'] + VOCABS['akshara_assamese']
VOCABS['all_bengali']   = VOCABS['english'] + VOCABS['unicode_bengali'] + VOCABS['bengali'] + VOCABS['ihtr_bengali'] + VOCABS['akshara_bengali'] 
VOCABS['all_bodo']      = VOCABS['english'] + VOCABS['unicode_bodo'] 
VOCABS['all_dogri']     = VOCABS['english'] + VOCABS['unicode_dogri']
VOCABS['all_gujarati']  = VOCABS['english'] + VOCABS['unicode_gujarati'] + VOCABS['gujarati'] + VOCABS['ihtr_gujarati'] + VOCABS['akshara_gujarati'] 
VOCABS['all_hindi']     = VOCABS['english'] + VOCABS['unicode_hindi'] + VOCABS['devanagari'] + VOCABS['ihtr_hindi'] + VOCABS['akshara_hindi']
VOCABS['all_kannada']   = VOCABS['english'] + VOCABS['unicode_kannada'] + VOCABS['kannada'] + VOCABS['ihtr_kannada'] + VOCABS['akshara_kannada'] 
VOCABS['all_kashmiri']  = VOCABS['english'] + VOCABS['unicode_kashmiri']
VOCABS['all_konkani']   = VOCABS['english'] + VOCABS['unicode_konkani']
VOCABS['all_maithili']  = VOCABS['english'] + VOCABS['unicode_maithili']
VOCABS['all_malayalam'] = VOCABS['english'] + VOCABS['unicode_malayalam'] + VOCABS['malayalam'] + VOCABS['ihtr_malayalam'] + VOCABS['akshara_malayalam'] 
VOCABS['all_manipuri']  = VOCABS['english'] + VOCABS['unicode_manipuri'] + VOCABS['akshara_manipuri']
VOCABS['all_marathi']   = VOCABS['english'] + VOCABS['unicode_marathi'] + VOCABS['akshara_marathi']
VOCABS['all_nepali']    = VOCABS['english'] + VOCABS['unicode_nepali']
VOCABS['all_odia']      = VOCABS['english'] + VOCABS['unicode_odia'] + VOCABS['odia'] + VOCABS['ihtr_odia'] + VOCABS['akshara_odia'] 
VOCABS['all_punjabi']   = VOCABS['english'] + VOCABS['unicode_punjabi'] + VOCABS['gurumukhi'] + VOCABS['ihtr_gurumukhi'] + VOCABS['akshara_gurumukhi'] 
VOCABS['all_sanskrit']  = VOCABS['english'] + VOCABS['unicode_sanskrit']
VOCABS['all_santali']   = VOCABS['english'] + VOCABS['unicode_santali']
VOCABS['all_sindhi']    = VOCABS['english'] + VOCABS['unicode_sindhi']
VOCABS['all_tamil']     = VOCABS['english'] + VOCABS['unicode_tamil'] + VOCABS['tamil'] + VOCABS['ihtr_tamil'] + VOCABS['akshara_tamil'] 
VOCABS['all_telugu']    = VOCABS['english'] + VOCABS['unicode_telugu'] + VOCABS['telugu'] + VOCABS['ihtr_telugu'] + VOCABS['akshara_telugu'] 
VOCABS['all_urdu']      = VOCABS['english'] + VOCABS['unicode_urdu'] + VOCABS['ihtr_urdu'] + VOCABS['akshara_urdu'] 


VOCABS['all_assamese']  = ''.join(sorted(list(set(VOCABS['all_assamese']))))
VOCABS['all_bengali']   = ''.join(sorted(list(set(VOCABS['all_bengali']))))
VOCABS['all_bodo']      = ''.join(sorted(list(set(VOCABS['all_bodo']))))
VOCABS['all_dogri']     = ''.join(sorted(list(set(VOCABS['all_dogri']))))
VOCABS['all_gujarati']  = ''.join(sorted(list(set(VOCABS['all_gujarati']))))
VOCABS['all_hindi']     = ''.join(sorted(list(set(VOCABS['all_hindi']))))
VOCABS['all_kannada']   = ''.join(sorted(list(set(VOCABS['all_kannada']))))
VOCABS['all_kashmiri']  = ''.join(sorted(list(set(VOCABS['all_kashmiri']))))
VOCABS['all_konkani']   = ''.join(sorted(list(set(VOCABS['all_konkani']))))
VOCABS['all_maithili']  = ''.join(sorted(list(set(VOCABS['all_maithili']))))
VOCABS['all_malayalam'] = ''.join(sorted(list(set(VOCABS['all_malayalam']))))
VOCABS['all_manipuri']  = ''.join(sorted(list(set(VOCABS['all_manipuri']))))
VOCABS['all_marathi']   = ''.join(sorted(list(set(VOCABS['all_marathi']))))
VOCABS['all_nepali']    = ''.join(sorted(list(set(VOCABS['all_nepali']))))
VOCABS['all_odia']      = ''.join(sorted(list(set(VOCABS['all_odia']))))
VOCABS['all_punjabi']   = ''.join(sorted(list(set(VOCABS['all_punjabi']))))
VOCABS['all_sanskrit']  = ''.join(sorted(list(set(VOCABS['all_sanskrit']))))
VOCABS['all_santali']   = ''.join(sorted(list(set(VOCABS['all_santali']))))
VOCABS['all_sindhi']    = ''.join(sorted(list(set(VOCABS['all_sindhi']))))
VOCABS['all_tamil']     = ''.join(sorted(list(set(VOCABS['all_tamil']))))
VOCABS['all_telugu']    = ''.join(sorted(list(set(VOCABS['all_telugu']))))
VOCABS['all_urdu']      = ''.join(sorted(list(set(VOCABS['all_urdu']))))



VOCABS['indic'] = "".join(
    dict.fromkeys(
        VOCABS['english'] +
        VOCABS['all_assamese'] +
        VOCABS['all_bengali'] +
        VOCABS['all_bodo'] +
        VOCABS['all_dogri'] +
        VOCABS['all_gujarati'] +
        VOCABS['all_hindi'] +
        VOCABS['all_kannada'] +
        VOCABS['all_kashmiri'] +
        VOCABS['all_konkani'] +
        VOCABS['all_maithili'] +
        VOCABS['all_malayalam'] +
        VOCABS['all_manipuri'] +
        VOCABS['all_marathi'] +
        VOCABS['all_nepali'] +
        VOCABS['all_odia'] +
        VOCABS['all_punjabi'] +
        VOCABS['all_sanskrit'] +
        VOCABS['all_santali'] +
        VOCABS['all_sindhi'] +
        VOCABS['all_tamil'] +
        VOCABS['all_telugu'] +
        VOCABS['all_urdu']
    )
)

VOCABS['indic'] = ''.join(sorted(list(set(VOCABS['indic']))))
