import json
from collections import OrderedDict

def proceseaza_si_scrie_date(cale_fisier1, cale_fisier2, cale_fisier_iesire):
    data = []
    for cale_fisier in [cale_fisier1, cale_fisier2]:
        with open(cale_fisier, 'r', encoding='utf-8') as fisier:
            data.extend(json.load(fisier))
    linkuri_unice = OrderedDict()

    for element in data:
        link_ref = element.get('Link_Ref', '')
        if link_ref not in linkuri_unice:
            linkuri_unice[link_ref] = f'"{link_ref}","{element["Titlu_Ref_Clean"]}","{element["Titlu_link_Ref"]}"\n'

    with open(cale_fisier_iesire, 'w', encoding='utf-8') as fisier_iesire:
        for linie in linkuri_unice.values():
            fisier_iesire.write(linie)

if __name__ == '__main__':

    ''' De modificat caile'''
    trainingFile = '../all_datas_training_v2.json'
    testFile = '../all_datas_test_v2.json'
    cale_fisier_iesire = 'entities.csv'
    proceseaza_si_scrie_date(trainingFile, testFile, cale_fisier_iesire)

