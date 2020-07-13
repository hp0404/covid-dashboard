import os
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import chain
from collections import Counter


URL = "https://raw.githubusercontent.com/VasiaPiven/covid19_ua/master/covid19_by_area_type_hosp_dynamics.csv"
TODAY = datetime.today().strftime("%Y-%m-%d")
CURRENT_LOCATION = os.path.dirname(os.path.abspath(__file__))
LIKARNI_LOCATION = os.path.join(CURRENT_LOCATION, "data")
OUTPUTS_LOCATION = os.path.join(CURRENT_LOCATION, "data", "outputs")
SAVE_FILE = f"{OUTPUTS_LOCATION}/monitoring_v5_{TODAY}.csv"


def _metadata(s) -> str:
    """ pd.Series[List[List]] -> str 
    
    
    Examples
    --------
    >>> df["person_gender"].apply(_metadata)
    
    -> [[Жіноча, Жіноча], [Жіноча, Жіноча, Жіноча], [...
    -> ((Жіноча, 8), (Чоловіча, 2))
    -> Жіноча: 8, Чоловіча: 2
    """
    
    return ", ".join(f"{k}: {v}" for k, v in dict(Counter(chain(*s))).items())


def _pending_susp(table, offset_days: int = 2):
    """ Розраховує активні підозри.
    
    Сума нових підозр кожної окремої лікарні за звітній період + два попередніх дні.  
    """

    df = table.sort_values("zvit_date").copy()
    
    for date in df["zvit_date"].unique():
        
        current_date = pd.to_datetime(date)
        previous_date = current_date - pd.DateOffset(days=offset_days)
        tmp = df.set_index("zvit_date")[previous_date:current_date].copy()

        g = tmp.groupby("edrpou_hosp_tmp", as_index=False)["new_susp"].sum()
        g = g.rename(columns={"new_susp": "pending_susp"})
        g["zvit_date"] = current_date

        yield g


def agg(table):
    """ Агрегує дані за такою схемою: 
    один категорія - один рядок -> одна лікарня - один рядок.
        
    
    Examples
    --------
    >>> df.head()
        zvit_date   edrpou_hosp   person_gender            is_medical_worker  active_confirm
    0   2020-04-26  36338715      Жіноча                   Ні                 4
    1   2020-04-26  36338715      Жіноча                   Так                1
    2   2020-04-26  36338715      Жіноча                   Ні                 2
    3   2020-04-26  36338715      Чоловіча                 Ні                 3
    4   2020-04-26  36338715      Чоловіча                 Ні                 1
    5   2020-04-26  36338715      Чоловіча                 Ні                 2
    6   2020-04-26  36338715      Чоловіча                 Ні                 1
    7   2020-04-26  36338715      Жіноча                   Ні                 1
    8   2020-04-25  36338715      Чоловіча                 Ні                 3
    9   2020-04-25  36338715      Чоловіча                 Ні                 1

    
    >>> agg(df)
        zvit_date   edrpou_hosp   person_gender            is_medical_worker  active_confirm
    0   2020-04-26  36338715      Жіноча: 8, Чоловіча: 7   Ні: 14, Так: 1     15
    1   2020-04-25  36338715      Жіноча: 8, Чоловіча: 8   Ні: 15, Так: 1     16
    2   2020-04-24  36338715      Жіноча: 8, Чоловіча: 8   Ні: 15, Так: 1     16
    """

    df = table.copy()
    #df["zvit_date"] = pd.to_datetime(df["zvit_date"])
    df.loc[df["zvit_date"] == "2002-05-21", "zvit_date"] = "2020-05-21"
    df.loc[df["zvit_date"] == "2002-05-22", "zvit_date"] = "2020-05-22"
    df["zvit_date"] = pd.to_datetime(df["zvit_date"])
    
    df["edrpou_hosp_tmp"] = np.where(
        df["edrpou_hosp"].eq("Самоізоляція"),
        df["registration_area"] + "_" + df["edrpou_hosp"],
        df["edrpou_hosp"],
    )
    
    df["gender_active"] = (
        df["person_gender"]
        .str.cat(df["active_confirm"].astype(str), sep=";")
        .str.split(";")
    )
    df["age_active"] = (
        df["person_age_group"]
        .str.cat(df["active_confirm"].astype(str), sep=";")
        .str.split(";")
    )
    df["add_conditions_active"] = (
        df["add_conditions"]
        .str.cat(df["active_confirm"].astype(str), sep=";")
        .str.split(";")
    )
    df["is_medical_worker_active"] = (
        df["is_medical_worker"]
        .str.cat(df["active_confirm"].astype(str), sep=";")
        .str.split(";")
    )
    
    df["person_gender"] = [
        [category] * int(count) for category, count in df["gender_active"]
    ]
    df["person_age_group"] = [
        [category] * int(count) for category, count in df["age_active"]
    ]
    df["add_conditions"] = [
        [category] * int(count) for category, count in df["add_conditions_active"]
    ]
    df["is_medical_worker"] = [
        [category] * int(count) for category, count in df["is_medical_worker_active"]
    ]

    g = (
        df.groupby(["zvit_date", "edrpou_hosp_tmp"])
        .agg(
            registration_area=("registration_area", "first"),
            total_area=("priority_hosp_area", "first"),
            edrpou_hosp=("edrpou_hosp", "first"),
            legal_entity_name_hosp=("legal_entity_name_hosp", "first"),
            lat=("legal_entity_lat", "first"),
            lng=("legal_entity_lng", "first"),
            person_gender=("person_gender", list),
            person_age_group=("person_age_group", list),
            add_conditions=("add_conditions", list),
            is_medical_worker=("is_medical_worker", list),
            new_susp=("new_susp", "sum"),
            new_confirm=("new_confirm", "sum"),
            active_confirm=("active_confirm", "sum"),
            new_death=("new_death", "sum"),
            new_recover=("new_recover", "sum"),
        )
        .reset_index()
    )

    g["person_gender"] = g["person_gender"].apply(_metadata)
    g["person_age_group"] = g["person_age_group"].apply(_metadata)
    g["add_conditions"] = g["add_conditions"].apply(_metadata)
    g["is_medical_worker"] = g["is_medical_worker"].apply(_metadata)

    return g


def filling_inactive(table: pd.DataFrame) -> pd.DataFrame:
    """ Додає "неактивні" лікарні.
    
    Вихідна таблиця містить лише "активні" лікарні: 
    якщо за звітній період у лікарні немає активних випадків - підозр, хворих тощо, - 
    вона не вноситься в цю таблицю і відповідно зникає з дашборду. 
    
    Для коректного відображення лікарень на дашборді ф-ція додає всі "неактивні" лікарні
    кожного наступного дня, коли активних випадків у цій лікарні вже не зафіксовано. 
    """

    _COLS = [
        "zvit_date",
        "edrpou_hosp_tmp",
        "new_susp",
        "new_confirm",
        "active_confirm",
        "new_death",
        "new_recover",
    ]

    df_tmp = table.assign(_value=table[_COLS[2:]].sum(axis=1))
    df_hsp = table.assign(_value=table[_COLS].sum(axis=1))

    piv = df_hsp[["zvit_date", "edrpou_hosp_tmp", "_value"]].pivot(
        "zvit_date", "edrpou_hosp_tmp", "_value"
    )

    meta_filled = (
        piv.mask(piv.ffill().notna() & piv.isna(), 0)
        .stack()
        .astype(int)
        .reset_index(name="_value")
    )

    merged = pd.merge(
        meta_filled, df_tmp, 
        how="left", 
        on=["zvit_date", "edrpou_hosp_tmp", "_value"]
    )

    merged["filter"] = np.where(merged["_value"].eq(0), "Додати метадані", "ОК")
    ok = merged.loc[merged["filter"].eq("ОК")].copy()
    ne_ok = merged.loc[merged["filter"].ne("ОК")].copy()

    data = table.drop_duplicates("edrpou_hosp_tmp").loc[:, "edrpou_hosp_tmp":].copy()
    incomplete_filled = pd.merge(
        ne_ok[_COLS[:2]], data, 
        how="left",
        on=["edrpou_hosp_tmp"]
    )

    incomplete_filled["person_gender"] = ""
    incomplete_filled["person_age_group"] = ""
    incomplete_filled["add_conditions"] = ""
    incomplete_filled["is_medical_worker"] = ""
    incomplete_filled.loc[:, "new_susp":"new_recover"] = 0

    result = (
        pd.concat([ok, incomplete_filled], ignore_index=True)
        .sort_values("zvit_date")
    )

    result.loc[:, "new_susp":"new_recover"] = result.loc[
        :, "new_susp":"new_recover"
    ].astype(int)

    return result.drop(["filter", "_value"], 1)


def merge_pending(table: pd.DataFrame) -> pd.DataFrame:
    """ Приєднує активні підозри до основної таблиці. """

    pending_susp = pd.concat(_pending_susp(table))
    result = pd.merge(
        table, pending_susp, 
        how="left",
        on=["zvit_date", "edrpou_hosp_tmp"]
    )

    return result


def filling_unused(table: pd.DataFrame, drop_metadata: bool = True) -> pd.DataFrame:
    """ Додає лікарні 1 хвилі. """

    _DTYPES = {"Код ЄДРПОУ": str, "edrpou_hosp": str}
    _LIKARNI_COLS = {
        "Код ЄДРПОУ": "edrpou_hosp",
        "Назва закладу": "legal_entity_name_hosp",
        "Область": "total_area",
        "Коорд. Х": "lat",
        "Коорд. Y": "lng",
    }
    
    _META_COLS = [
        "person_gender", "person_age_group", 
        "add_conditions", "is_medical_worker"
    ]

    hosp = pd.read_excel(
        f"{LIKARNI_LOCATION}/01_zoz_list240_v2_addresses.xlsx", dtype=_DTYPES
        )

    product = pd.MultiIndex.from_product(
        iterables=[table["zvit_date"].unique(), hosp["Код ЄДРПОУ"].unique()],
        names=["zvit_date", "edrpou_hosp"],
    ).to_frame(index=False)

    complete_hosp = pd.merge(
        product,
        hosp.rename(columns=_LIKARNI_COLS)[list(_LIKARNI_COLS.values())],
        how="left",
        on="edrpou_hosp",
    )

    for col in table.columns:
        if col not in complete_hosp.columns:
            complete_hosp[col] = 0

    complete_hosp["zvit_date"] = pd.to_datetime(complete_hosp["zvit_date"])
    complete_hosp["registration_area"] = complete_hosp["total_area"]
    complete_hosp["person_gender"] = ""
    complete_hosp["person_age_group"] = ""
    complete_hosp["add_conditions"] = ""
    complete_hosp["is_medical_worker"] = ""

    result = pd.concat([table, complete_hosp], ignore_index=True).fillna(0)

    result["total_area"] = result["total_area"].str.replace("^Київ$", "м. Київ")
    result["_val"] = result.loc[:, "new_susp":"new_recover"].sum(axis=1)

    result = (
        result.sort_values("_val")
        .drop_duplicates(["zvit_date", "total_area", "edrpou_hosp"], keep="last")
        .drop(["_val", "edrpou_hosp_tmp"], 1)
    )
    
    return result.drop(_META_COLS, 1) if drop_metadata else result


def make_dataset():

    df = pd.read_csv(URL)
    df.loc[df["zvit_date"] == "2002-05-22", "zvit_date"] = "2020-05-22"
    df.loc[df["zvit_date"] == "2010-05-24", "zvit_date"] = "2020-05-24"
    aggregated = agg(df)
    active_hospitals = filling_inactive(aggregated)
    pending_calc = merge_pending(active_hospitals)
    complete_table = filling_unused(pending_calc, drop_metadata=False)
    complete_table["total_area"].str.replace("^полтавська", "Полтавська")

    complete_table["lat"] = (
        complete_table["lat"].astype(str)
        .str.replace(",", ".").astype(float)
    )

    complete_table["lng"] = (
        complete_table["lng"].astype(str)
        .str.replace(",", ".").astype(float)
    )

    print("total confirmed cases", complete_table["new_confirm"].sum())
    print("total confirmed deaths", complete_table["new_death"].sum())
    print("total confirmed recoveries", complete_table["new_recover"].sum())

    complete_table.to_csv(
            SAVE_FILE, sep=";", index=False
        )


if __name__ == "__main__":
    
    if not os.path.isfile(SAVE_FILE):
        make_dataset()
    