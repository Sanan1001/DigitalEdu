#создай здесь свой индивидуальный проект!
import pandas as pd
df = pd.read_csv('train.csv')
df.dropna()
df.drop(['id','bdate','graduation','has_photo','people_main', 'relation', 'has_mobile', 'city', 'followers_count', 'education_form', 'career_start', 'career_end', 'occupation_name', 'last_seen'],axis = 1, inplace = True)

def sex_apply(sex):
    if sex == 2:
        return 1
    return 0

def edu_stat_apply(education_status):
    if education_status == 'Undergraduate applicant':
        return 0
    if education_status == "Student (Bachelor's)" or education_status == "Student (Specialist)" or education_status == "Student (Master's)" :
        return 1
    if education_status == 'Alumnus (Specialist)' or education_status == "Alumnus (Bachelor's)" or education_status == "Alumnus (Master's)":
        return 2
    if education_status == 'PhD' or education_status == "Candidate of Sciences":
        return 3

df['education_status'] = df['education_status'].apply(edu_stat_apply)

# def fill_sex(sex):
#     if sex == 'male':
#         return 1
#     return 0

df['sex'] = df['sex'].apply(sex_apply)

# def education_status(education_form):

print(df['sex'].value_counts())
# print(df['langs'].value_counts())

def langs_apply(langs):
    if langs.find('Русский') != -1:
        return 0
    return 1

df['langs'] = df['langs'].apply(langs_apply)
print(df['langs'].value_counts())

def fill_occupation_type(row):
    if row['education_status'] == 1:
        return 'university'
    return 'work'

df['occupation_type'].fillna('university', inplace = True)

def occupation_type_apply(occupation_type):
    if occupation_type == 'university':
        return 1 
    return 0
df['occupation_type'] = df['occupation_type'].apply(occupation_type_apply)
print(df['occupation_type'].value_counts())
print(df['education_status'].value_counts())

print(df['life_main'].value_counts())

def life_main_apply(life_main):
    if life_main == 'False' or life_main == '0':
        return 0
    if life_main == '1':    #1 — семья и дети; 2 — карьера и деньги;        
        return 1
    if life_main == '2':    #3 — развлечения и отдых; 4 — наука и исследования;
        return 2 
    if life_main == '3':    #5 — совершенствование мира; 6 — саморазвитие;
        return 3
    if life_main == '4':    #7 — красота и искусство; 8 — слава и влияние).
        return 4
    if life_main == '5':
        return 5
    if life_main == '6':
        return 6
    if life_main == '7':
        return 7
    if life_main == '8':
        return 8                                            

df['life_main'] = df['life_main'].apply(life_main_apply)
print(df['life_main'].value_counts())

life_m1 = 0
life_m2 = 0
life_m3 = 0
life_m4 = 0
life_m5 = 0
life_m6 = 0
life_m7 = 0
life_m8 = 0

rus_lang = 0
oth_lang = 0

def hz(row):
    global life_m1, life_m2, life_m3,life_m4,life_m5,life_m6,life_m7,life_m8,rus_lang, oth_lang
    if row['result'] == 1:
        if row['life_main'] == 1:
                life_m1 += 1
        if row['life_main'] == 2:
                life_m2 += 1
        if row['life_main'] == 3:
                life_m3 += 1
        if row['life_main'] == 4:
                life_m4 += 1
        if row['life_main'] == 5:
                life_m5 += 1
        if row['life_main'] == 6:
                life_m6 += 1
        if row['life_main'] == 7:
                life_m7 += 1
        if row['life_main'] == 8:
                life_m8 += 1        
        if row['langs'] == 1:
            rus_lang += 1
        else:
            oth_lang += 1
    return False

df['life_main'] = df.apply(hz, axis = 1)
df['langs'] = df.apply(hz, axis = 1)
print(life_m1, life_m2,life_m3,life_m4,life_m5,life_m6,life_m7,life_m8)

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

x = df.drop('result', axis = 1)
y = df['result']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

classifier = KNeighborsClassifier( n_neighbors = 5 )
classifier.fit( x_train, y_train )

y_pred = classifier.predict(x_test)

percent = accuracy_score(y_test, y_pred) * 100
print('Процент правильно предсказанных исходов:',round( percent, 2))
print('Люди знающие русский:', rus_lang)
print('Люди знающие другие языки:', oth_lang)
print('Больше всего людей купивших курс (',life_m6,') выбрали главным в жизни - саморазвитие ')

# df.info()