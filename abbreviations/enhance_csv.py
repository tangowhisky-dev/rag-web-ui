#!/usr/bin/env python3
"""Enhance abbreviations.csv based on the 7 rules for RAG use case.

Rules applied:
1. Combination abbreviations (fd coy = Field Company)
2. Compound words (discont = Discontinue)
3. Verb forms: ALL forms of a verb share the same abbreviation
4. Prefix combinations: A=Assistant, D=Deputy, DA=Deputy Assistant
5. Plurals: Add 's' to abbreviation for plural forms
6. Capital letters: Preserve as-is
7. Derivatives: d, ed, ment, ing, al, etc.
"""
import csv
from collections import defaultdict

CSV_IN = "abbreviations.csv"
CSV_OUT = "abbreviations_enhanced.csv"


def load_csv(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


# === RULE 3 & 7: Complete verb/derivative forms ===
# For each abbreviation that has a verb base form, add ALL missing standard forms.
# Format: abbreviation -> [list of (form, category) to add if not present]
# Only includes forms that are linguistically correct and not already in CSV.

VERB_FORMS = {
    # === Already partially expanded verbs (add missing forms) ===
    'wdr': [
        ('Withdrew', 'general'), ('Withdraws', 'general'), ('Withdrawing', 'general'),
    ],
    'sp': [
        ('Supports', 'general'), ('Supporter', 'general'), ('Supportive', 'general'),
    ],
    'ack': [
        ('Acknowledges', 'general'), ('Acknowledging', 'general'),
    ],
    'adv': [
        ('Advances', 'general'), ('Advancement', 'general'),
    ],
    'coord': [
        ('Coordinates', 'general'), ('Co-ordinates', 'general'),
        ('Co-ordinated', 'general'), ('Co-ordinating', 'general'),
        ('Co-ordinator', 'general'),
    ],
    'dep': [
        ('Departs', 'general'), ('Departing', 'general'),
    ],
    'dir': [
        ('Directs', 'general'), ('Directors', 'general'),
    ],
    'eff': [
        ('Effects', 'general'), ('Effecting', 'general'),
    ],
    'incl': [
        ('Inclusion', 'general'),
    ],
    'loc': [
        ('Locates', 'general'), ('Locations', 'general'),
    ],
    'op': [
        ('Operates', 'general'), ('Operating', 'general'),
        ('Operations', 'general'), ('Operators', 'general'),
    ],
    'rec': [
        ('Recovers', 'general'), ('Recovering', 'general'), ('Recoveries', 'general'),
    ],
    'rep': [
        ('Represents', 'general'), ('Representing', 'general'),
        ('Representation', 'general'), ('Representatives', 'general'),
    ],
    'req': [
        ('Requires', 'general'), ('Requiring', 'general'), ('Requirements', 'general'),
    ],
    'recce': [
        ('Reconnoitres', 'general'), ('Reconnoitring', 'general'),
    ],

    # === Verbs with some forms but missing many ===
    'aloc': [('Allocates', 'general'), ('Allocating', 'general')],
    'apch': [('Approaches', 'general')],
    'appl': [('Applies', 'general'), ('Applying', 'general')],
    'appt': [('Appoints', 'general'), ('Appointing', 'general'), ('Appointments', 'general')],
    'aprc': [('Appreciates', 'general'), ('Appreciating', 'general')],
    'arng': [('Arranges', 'general')],
    'arr': [('Arrives', 'general')],
    'asg': [('Assigns', 'general')],
    'aslt': [('Assaults', 'general')],
    'asst': [('Assists', 'general'), ('Assisting', 'general')],
    'att': [('Attaches', 'general'), ('Attaching', 'general')],
    'attk': [('Attacks', 'general')],
    'audt': [('Audits', 'general'), ('Auditing', 'general')],
    'auth': [('Authorises', 'general'), ('Authorising', 'general')],
    'bkld': [('Backloads', 'general')],
    'cal': [('Calibrates', 'general'), ('Calibrated', 'general'), ('Calibrating', 'general')],
    'calc': [('Calculates', 'general'), ('Calculating', 'general')],
    'cam': [('Camouflages', 'general'), ('Camouflaging', 'general')],
    'cfm': [('Confirms', 'general'), ('Confirming', 'general')],
    'ck': [('Cooks', 'general')],
    'cmt': [('Commits', 'general'), ('Committed', 'general'), ('Committing', 'general')],
    'comb': [('Combines', 'general'), ('Combining', 'general')],
    'comd': [('Commands', 'general')],
    'comm': [('Communicates', 'general'), ('Communicating', 'general')],
    'con': [('Controls', 'general'), ('Controlling', 'general')],
    'conc': [('Concentrates', 'general'), ('Concentrating', 'general')],
    'concl': [('Concludes', 'general'), ('Concluding', 'general')],
    'consd': [('Considers', 'general')],
    'const': [('Constructs', 'general'), ('Constructing', 'general')],
    'cont': [('Continues', 'general'), ('Continuing', 'general')],
    'contam': [('Contaminates', 'general'), ('Contaminating', 'general')],
    'coop': [('Cooperates', 'general'), ('Cooperating', 'general')],
    'cpt': [('Computes', 'general')],
    'ctr sign': [('Countersigns', 'general')],
    'cvy': [('Conveys', 'general')],
    'dec': [('Decreases', 'general'), ('Decreasing', 'general')],
    'decen': [('Decentralises', 'general')],
    'decontam': [('Decontaminates', 'general'), ('Decontaminating', 'general')],
    'def': [('Defends', 'general'), ('Defending', 'general')],
    'del': [('Delivers', 'general'), ('Delivering', 'general')],
    'demo': [('Demonstrates', 'general'), ('Demonstrating', 'general')],
    'depl': [('Deploys', 'general'), ('Deploying', 'general')],
    'desp': [('Despatches', 'general')],
    'det': [('Detaches', 'general'), ('Detaching', 'general')],
    'dev': [('Develops', 'general'), ('Developing', 'general')],
    'disch': [('Discharges', 'general'), ('Discharging', 'general')],
    'distr': [('Distributes', 'general'), ('Distributing', 'general')],
    'dml': [('Demolishes', 'general'), ('Demolishing', 'general')],
    'emb': [('Embarks', 'general'), ('Embarking', 'general')],
    'emp': [('Employs', 'general'), ('Employing', 'general')],
    'encl': [('Encloses', 'general'), ('Enclosing', 'general')],
    'enrl': [('Enrols', 'general'), ('Enrolling', 'general')],
    'est': [('Estimates', 'general'), ('Estimating', 'general')],
    'estb': [('Establishes', 'general'), ('Establishing', 'general')],
    'evac': [('Evacuates', 'general'), ('Evacuating', 'general')],
    'eval': [('Evaluates', 'general')],
    'ex': [('Exercises', 'general'), ('Exercising', 'general')],
    'exam': [('Examines', 'general'), ('Examining', 'general')],
    'excl': [('Excludes', 'general')],
    'exec': [('Executes', 'general'), ('Executing', 'general')],
    'expd': [('Expedites', 'general'), ('Expediting', 'general')],
    'expl': [('Explodes', 'general'), ('Exploding', 'general')],
    'fol': [('Follows', 'general')],
    'gd': [('Guards', 'general')],
    'ident': [('Identifies', 'general'), ('Identifying', 'general')],
    'ill': [('Illuminates', 'general')],
    'inc': [('Increases', 'general'), ('Increasing', 'general')],
    'infil': [('Infiltrates', 'general')],
    'info': [('Informs', 'general'), ('Informing', 'general')],
    'insp': [('Inspects', 'general')],
    'instl': [('Installs', 'general'), ('Installing', 'general')],
    'instr': [('Instructs', 'general'), ('Instructing', 'general')],
    'intd': [('Interdicts', 'general')],
    'intr': [('Interrogates', 'general'), ('Interrogating', 'general')],
    'intro': [('Introduces', 'general')],
    'maint': [('Maintains', 'general'), ('Maintaining', 'general')],
    'mob': [('Mobilized', 'general'), ('Mobilizes', 'general'), ('Mobilizing', 'general')],
    'mov': [('Moves', 'general'), ('Moving', 'general')],
    'nav': [('Navigates', 'general'), ('Navigating', 'general')],
    'neut': [('Neutralises', 'general'), ('Neutralising', 'general')],
    'org': [('Organises', 'general'), ('Organising', 'general')],
    'pen': [('Penetrates', 'general'), ('Penetrating', 'general')],
    'prep': [('Prepares', 'general'), ('Preparing', 'general')],
    'progm': [('Programmes', 'general')],
    'proj': [('Projects', 'general'), ('Projecting', 'general')],
    'prom': [('Promotes', 'general'), ('Promoting', 'general')],
    'ptl': [('Patrols', 'general')],
    'pub': [('Publishes', 'general'), ('Publishing', 'general')],
    'rect': [('Recruits', 'general')],
    'ref': [('Referring', 'general')],
    'reg': [('Regulates', 'general'), ('Regulating', 'general')],
    'rel': [
        ('Releases', 'general'), ('Releasing', 'general'),
        ('Relieves', 'general'), ('Relieving', 'general'),
    ],
    'reorg': [('Reorganises', 'general'), ('Reorganising', 'general')],
    'rft': [('Reinforces', 'general'), ('Reinforcing', 'general'), ('Reinforced', 'general')],
    'rpl': [('Replenishes', 'general'), ('Replenishing', 'general'), ('Replenished', 'general')],
    'rvt': [('Reverts', 'general'), ('Reverting', 'general')],
    'sal': [('Salvages', 'general'), ('Salvaging', 'general')],
    'sel': [('Selects', 'general'), ('Selecting', 'general')],
    'sta': [('Stations', 'general')],
    'std': [('Standardises', 'general'), ('Standardising', 'general')],
    'str': [('Strengthens', 'general'), ('Strengthening', 'general')],
    'svc': [('Services', 'general'), ('Servicing', 'general')],
    'sync': [('Synchronised', 'general'), ('Synchronises', 'general'), ('Synchronising', 'general')],
    'tfr': [('Transfers', 'general'), ('Transferring', 'general'), ('Transferred', 'general')],
    'tpt': [('Transports', 'general'), ('Transporting', 'general')],
    'tx': [('Transmits', 'general')],

    # === Single-form verbs that need ALL forms ===
    'spk': [('Spoke', 'general'), ('Speaks', 'general'), ('Speaking', 'general'), ('Spoken', 'general')],
    'discont': [('Discontinues', 'general'), ('Discontinued', 'general'), ('Discontinuing', 'general')],
}

# === RULE 5: Plural forms ===
# (singular_abbr, singular_form, plural_form, category)
# Skip abbreviations that already work for both singular and plural (HQ, sig, cas)
PLURAL_FORMS = [
    # Military ranks and appointments
    ('CO', 'Commanding Officer', 'Commanding Officers', 'RANKS'),
    ('MO', 'Medical Officer', 'Medical Officers', 'RANKS'),
    ('offr', 'Officer', 'Officers', 'general'),
    ('Gp', 'Group', 'Groups', 'general'),
    ('Gp Capt', 'Group Captain', 'Group Captains', 'RANKS'),
    ('bn', 'Battalion', 'Battalions', 'general'),
    ('coy', 'Company', 'Companies', 'general'),
    ('sqn', 'Squadron', 'Squadrons', 'general'),
    ('bde', 'Brigade', 'Brigades', 'general'),
    ('div', 'Division', 'Divisions', 'general'),
    ('regt', 'Regiment', 'Regiments', 'general'),
    ('fd', 'Field', 'Fields', 'general'),
    ('Arty', 'Artillery', 'Artilleries', 'general'),
    ('engr', 'Engineer', 'Engineers', 'general'),
    ('Sig', 'Signal', 'Signals', 'general'),
    ('Sqn Ldr', 'Squadron Leader', 'Squadron Leaders', 'RANKS'),
    ('Flt Lt', 'Flight Lieutenant', 'Flight Lieutenants', 'RANKS'),
    ('Fg Offr', 'Flying Officer', 'Flying Officers', 'RANKS'),
    ('Wg Cdr', 'Wing Commander', 'Wing Commanders', 'RANKS'),
    ('Air Cdr', 'Air Commodore', 'Air Commodores', 'RANKS'),
    ('AVM', 'Air Vice Marshal', 'Air Vice Marshals', 'RANKS'),
    ('2 Lt', 'Second Lieutenant', 'Second Lieutenants', 'RANKS'),
    ('Lt', 'Lieutenant', 'Lieutenants', 'RANKS'),
    ('Capt', 'Captain', 'Captains', 'RANKS'),
    ('Maj', 'Major', 'Majors', 'RANKS'),
    ('Lt Col', 'Lieutenant Colonel', 'Lieutenant Colonels', 'RANKS'),
    ('Col', 'Colonel', 'Colonels', 'RANKS'),
    ('Brig', 'Brigadier', 'Brigadiers', 'RANKS'),
    ('Maj Gen', 'Major General', 'Major Generals', 'RANKS'),
    ('Lt Gen', 'Lieutenant General', 'Lieutenant Generals', 'RANKS'),
    ('Gen', 'General', 'Generals', 'RANKS'),
    ('WO', 'Warrant Officer', 'Warrant Officers', 'RANKS'),
    ('NCO', 'Non-Commissioned Officer', 'Non-Commissioned Officers', 'RANKS'),
    ('JCO', 'Junior Commissioned Officer', 'Junior Commissioned Officers', 'RANKS'),
    ('adjt', 'Adjutant', 'Adjutants', 'STAFF AND APPTS'),
    ('QMG', 'Quartermaster General', 'Quartermasters General', 'STAFF AND APPTS'),
    ('OC', 'Officer Commanding', 'Officers Commanding', 'STAFF AND APPTS'),
    ('PO', 'Petty Officer', 'Petty Officers', 'RANKS'),
    ('CPO', 'Chief Petty Officer', 'Chief Petty Officers', 'RANKS'),
    ('LS', 'Leading Seaman', 'Leading Seamen', 'RANKS'),
    ('AB', 'Able Seaman', 'Able Seamen', 'RANKS'),
    ('Cpl', 'Corporal', 'Corporals', 'RANKS'),
    ('Sgt', 'Sergeant', 'Sergeants', 'RANKS'),
    ('SSgt', 'Staff Sergeant', 'Staff Sergeants', 'RANKS'),
    ('Pte', 'Private', 'Privates', 'RANKS'),
    ('LCpl', 'Lance Corporal', 'Lance Corporals', 'RANKS'),
    ('Rfn', 'Rifleman', 'Riflemen', 'RANKS'),
    ('Sapr', 'Sapper', 'Sappers', 'RANKS'),
    ('Dfr', 'Driver', 'Drivers', 'RANKS'),
    ('Gnr', 'Gunner', 'Gunners', 'RANKS'),
    ('Sig', 'Signalman', 'Signalmen', 'RANKS'),
    ('Cfn', 'Craftsman', 'Craftsmen', 'RANKS'),
    ('L/Nk', 'Lance Naik', 'Lance Naiks', 'RANKS'),
    ('Nk', 'Naik', 'Naiks', 'RANKS'),
    ('HK', 'Havildar', 'Havildars', 'RANKS'),
    ('SM', 'Subedar Major', 'Subedar Majors', 'RANKS'),
    # Units and formations
    ('tp', 'Troop', 'Troops', 'general'),
    ('tp', 'Troops', 'Troops', 'general'),  # already plural
    ('bty', 'Battery', 'Batteries', 'general'),
    ('spt', 'Squadron', 'Squadrons', 'general'),  # if spt = Squadron
    ('sec', 'Section', 'Sections', 'general'),
    ('pl', 'Platoon', 'Platoons', 'general'),
    ('fl', 'Flight', 'Flights', 'general'),
    ('flt', 'Flight', 'Flights', 'general'),
    ('wg', 'Wing', 'Wings', 'general'),
    ('gp', 'Group', 'Groups', 'general'),
    ('bde', 'Brigade', 'Brigades', 'general'),
    ('cdo', 'Commando', 'Commandos', 'general'),
    ('gren', 'Grenade', 'Grenades', 'general'),
    ('msl', 'Missile', 'Missiles', 'general'),
    ('mor', 'Mortar', 'Mortars', 'general'),
    ('tk', 'Tank', 'Tanks', 'general'),
    ('veh', 'Vehicle', 'Vehicles', 'general'),
    ('rkt', 'Rocket', 'Rockets', 'general'),
    ('mg', 'Machine gun', 'Machine guns', 'general'),
    ('MG', 'Machine-gun', 'Machine-guns', 'general'),
    ('msn', 'Mission', 'Missions', 'general'),
    ('tgt', 'Target', 'Targets', 'general'),
    ('obj', 'Objective', 'Objectives', 'general'),
    ('pos', 'Position', 'Positions', 'general'),
    ('loc', 'Location', 'Locations', 'general'),
    ('fd', 'Field', 'Fields', 'general'),
    ('wksp', 'Workshop', 'Workshops', 'general'),
    ('biv', 'Bivouac', 'Bivouacs', 'general'),
    ('har', 'Harbour', 'Harbours', 'general'),
    ('gar', 'Garrison', 'Garrisons', 'general'),
    ('pt', 'Point', 'Points', 'general'),
    ('st', 'Street', 'Streets', 'general'),
    ('vill', 'Village', 'Villages', 'general'),
    ('rly', 'Railway', 'Railways', 'general'),
    ('br', 'Bridge', 'Bridges', 'general'),
    ('rd', 'Road', 'Roads', 'general'),
    ('trk', 'Truck', 'Trucks', 'general'),
    ('msg', 'Message', 'Messages', 'general'),
    ('fig', 'Figure', 'Figures', 'general'),
    ('sch', 'Schedule', 'Schedules', 'general'),
    ('anx', 'Annex', 'Annexes', 'general'),
    ('appx', 'Appendix', 'Appendices', 'general'),
    ('ref', 'Reference', 'References', 'general'),
    ('memo', 'Memorandum', 'Memoranda', 'general'),
    ('docu', 'Document', 'Documents', 'general'),
    ('freq', 'Frequency', 'Frequencies', 'general'),
    ('alt', 'Altitude', 'Altitudes', 'general'),
    ('az', 'Azimuth', 'Azimuths', 'general'),
    ('deg', 'Degree', 'Degrees', 'general'),
    ('wt', 'Weight', 'Weights', 'general'),
    ('ht', 'Height', 'Heights', 'general'),
    ('vol', 'Volume', 'Volumes', 'general'),
    ('sq', 'Square', 'Squares', 'general'),
    ('cm', 'Centimetre', 'Centimetres', 'general'),
    ('km', 'Kilometre', 'Kilometres', 'general'),
    ('mm', 'Millimetre', 'Millimetres', 'general'),
    ('in', 'Inch', 'Inches', 'general'),
    ('yd', 'Yard', 'Yards', 'general'),
    ('gal', 'Gallon', 'Gallons', 'general'),
    ('lit', 'Litre', 'Litres', 'general'),
    ('kg', 'Kilogram', 'Kilograms', 'general'),
    ('mg', 'Milligram', 'Milligrams', 'general'),
    ('hr', 'Hour', 'Hours', 'general'),
    ('yr', 'Year', 'Years', 'general'),
    ('no', 'Number', 'Numbers', 'general'),
    ('pt', 'Point', 'Points', 'general'),
]

# === RULE 5: Abbreviations that already work for singular AND plural ===
ALREADY_PLURAL = [
    ('HQ', 'Headquarters', 'Headquarters', 'general'),
    ('sig', 'Signalman', 'Signalmen', 'general'),
    ('sig', 'Signals', 'Signals', 'general'),
    ('cas', 'Casualty', 'Casualties', 'general'),
    ('cas', 'Casualties', 'Casualties', 'general'),
    ('ammo', 'Ammunition', 'Ammunitions', 'ammunition'),
    ('recon', 'Reconnaissance', 'Reconnaissances', 'general'),
    ('GHQ', 'General Headquarters', 'General Headquarters', 'general'),
    ('AHQ', 'Air Headquarters', 'Air Headquarters', 'general'),
    ('NHQ', 'Naval Headquarters', 'Naval Headquarters', 'general'),
    ('pers', 'Personnel', 'Personnel', 'general'),
    ('obs', 'Obstacle', 'Obstacles', 'general'),
    ('obs', 'Obstacles', 'Obstacles', 'general'),
    ('res', 'Reserve', 'Reserves', 'general'),
    ('res', 'Reserves', 'Reserves', 'general'),
    ('en', 'Enemy', 'Enemies', 'general'),
]

# === RULE 4: Prefix combinations ===
PREFIX_ENTRIES = [
    ('A', 'Assistant', 'general'),
    ('D', 'Deputy', 'general'),
    ('DA', 'Deputy Assistant', 'general'),
    ('Dy', 'Deputy', 'general'),
    ('AME', 'Assistant Maintenance Engineer', 'STAFF AND APPTS'),
    ('DDMT', 'Deputy Director Military Training', 'STAFF AND APPTS'),
    ('DADS&T', 'Deputy Assistant Director of Supply and Transport', 'STAFF AND APPTS'),
    ('Dy MS', 'Deputy Military Secretary', 'STAFF AND APPTS'),
]

# === RULE 1-2: Combination and compound abbreviations ===
COMBINATION_ENTRIES = [
    ('fd coy', 'Field Company', 'general'),
    ('Fd Coy', 'Field Company', 'general'),
    ('discont', 'Discontinue', 'general'),
    ('M fd', 'Minefield', 'general'),
]


def main():
    rows = load_csv(CSV_IN)
    print(f"Loaded {len(rows)} rows from {CSV_IN}")

    # Build indexes
    by_abbr = defaultdict(list)
    by_abbr_ci = defaultdict(list)
    for r in rows:
        by_abbr[r['abbreviation']].append(r)
        by_abbr_ci[r['abbreviation'].lower()].append(r)

    existing_keys = set((r['abbreviation'], r['expanded_form']) for r in rows)
    new_rows = []

    def add_entry(abbr, form, cat):
        key = (abbr, form)
        if key not in existing_keys:
            new_rows.append({'abbreviation': abbr, 'expanded_form': form, 'category': cat})
            existing_keys.add(key)
            return True
        return False

    # === RULE 3 & 7: Add missing verb forms ===
    print("\n=== Rule 3 & 7: Adding missing verb forms ===")
    verb_added = 0
    for abbr, forms in VERB_FORMS.items():
        for form, cat in forms:
            if add_entry(abbr, form, cat):
                verb_added += 1
                print(f"  Added: {abbr} -> {form}")
    print(f"  Total verb forms added: {verb_added}")

    # === RULE 5: Add plural forms ===
    print("\n=== Rule 5: Adding plural forms ===")
    plural_added = 0
    for sing_abbr, sing_form, plural_form, cat in PLURAL_FORMS:
        # Find actual abbreviation (case-insensitive)
        actual_abbr = sing_abbr
        found = False
        for r in by_abbr.get(sing_abbr, []):
            if r['expanded_form'].lower() == sing_form.lower():
                actual_abbr = r['abbreviation']
                found = True
                break
        if not found:
            for r in by_abbr_ci.get(sing_abbr.lower(), []):
                if r['expanded_form'].lower() == sing_form.lower():
                    actual_abbr = r['abbreviation']
                    found = True
                    break
        if found:
            plural_abbr = actual_abbr + 's'
            if add_entry(plural_abbr, plural_form, cat):
                plural_added += 1
                print(f"  Added: {plural_abbr} -> {plural_form}")
        else:
            print(f"  SKIP (singular not found): {sing_abbr} -> {sing_form}")
    print(f"  Total plural forms added: {plural_added}")

    # === RULE 5: Add already-plural forms ===
    print("\n=== Rule 5: Adding already-plural forms ===")
    already_plural_added = 0
    for abbr, sing_form, plural_form, cat in ALREADY_PLURAL:
        if add_entry(abbr, plural_form, cat):
            already_plural_added += 1
            print(f"  Added: {abbr} -> {plural_form}")
    print(f"  Total already-plural forms added: {already_plural_added}")

    # === RULE 4: Add prefix entries ===
    print("\n=== Rule 4: Adding prefix combination entries ===")
    prefix_added = 0
    for abbr, form, cat in PREFIX_ENTRIES:
        if add_entry(abbr, form, cat):
            prefix_added += 1
            print(f"  Added: {abbr} -> {form}")
        else:
            print(f"  EXISTS: {abbr} -> {form}")
    print(f"  Total prefix entries added: {prefix_added}")

    # === RULE 1-2: Add combination/compound entries ===
    print("\n=== Rule 1-2: Adding combination/compound entries ===")
    combo_added = 0
    for abbr, form, cat in COMBINATION_ENTRIES:
        if add_entry(abbr, form, cat):
            combo_added += 1
            print(f"  Added: {abbr} -> {form}")
        else:
            print(f"  EXISTS: {abbr} -> {form}")
    print(f"  Total combination entries added: {combo_added}")

    # Combine and write
    all_rows = rows + new_rows
    all_rows.sort(key=lambda r: (r['abbreviation'].lower(), r['expanded_form'].lower()))

    with open(CSV_OUT, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['abbreviation', 'expanded_form', 'category'])
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    print(f"\n=== Summary ===")
    print(f"Original rows: {len(rows)}")
    print(f"New rows added: {len(new_rows)}")
    print(f"  - Verb forms: {verb_added}")
    print(f"  - Plural forms: {plural_added}")
    print(f"  - Already-plural forms: {already_plural_added}")
    print(f"  - Prefix entries: {prefix_added}")
    print(f"  - Combination entries: {combo_added}")
    print(f"Total rows: {len(all_rows)}")
    print(f"Unique abbreviations: {len(set(r['abbreviation'] for r in all_rows))}")
    print(f"Wrote to {CSV_OUT}")


if __name__ == "__main__":
    main()
