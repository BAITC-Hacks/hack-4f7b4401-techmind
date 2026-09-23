"""Career Quest. Python 3.10+, Streamlit, Pandas.
Install: python -m pip install "streamlit>=1.50,<2" "pandas>=2.2,<3"
Run: streamlit run app.py
All application data is stored in st.session_state, without a database.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st

HISTORY_COLUMNS = ["employee_id", "event_type", "status", "event_id", "completed_at"]
STATUSES = {"COMPLETED", "SKIPPED", "REJECTED"}
NEXT_GRADES = {"junior": "Middle", "middle": "Senior", "senior": "Lead"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def mock_data() -> dict[str, Any]:
    """Create independent demo objects for each browser session."""
    profiles = [
        ("EMP-001", "Backend Developer", "Middle", "Senior", 26,
         {"Python": 3, "System Design": 2, "SQL": 3, "Communication": 2, "Leadership": 1}),
        ("EMP-002", "Backend Developer", "Junior", "Middle", 8,
         {"Python": 2, "System Design": 1, "SQL": 1, "Communication": 2, "Leadership": 0}),
        ("EMP-003", "Data Analyst", "Middle", "Senior", 31,
         {"Python": 3, "SQL": 2, "Analytics": 3, "Communication": 3, "Leadership": 1}),
        ("EMP-004", "Backend Developer", "Senior", "Lead", 48,
         {"Python": 4, "System Design": 4, "SQL": 4, "Communication": 3, "Leadership": 2}),
        ("EMP-005", "Data Analyst", "Junior", "Middle", 11,
         {"Python": 1, "SQL": 2, "Analytics": 2, "Communication": 1, "Leadership": 0}),
        ("EMP-006", "Backend Developer", "Middle", "Senior", 19,
         {"Python": 4, "System Design": 3, "SQL": 2, "Communication": 2, "Leadership": 2}),
    ]
    employees = [dict(employee_id=eid, role=role, grade=grade, next_grade=target,
                      tenure_months=tenure, skills=skills)
                 for eid, role, grade, target, tenure, skills in profiles]
    activities = [
        ("EV-01", "Архитектура распределённых систем", "Hard Skills", "System Design", 1, 5),
        ("EV-02", "Проектирование API: практикум", "Hard Skills", "System Design", 1, 3),
        ("EV-03", "Python: от кода к production", "Hard Skills", "Python", 1, 5),
        ("EV-04", "Python: основы на практике", "Hard Skills", "Python", 1, 3),
        ("EV-05", "SQL: оптимизация сложных запросов", "Hard Skills", "SQL", 1.5, 5),
        ("EV-06", "SQL: аналитический тренажёр", "Hard Skills", "SQL", 1, 3),
        ("EV-07", "Обратная связь без конфликтов", "Soft Skills", "Communication", 1, 4),
        ("EV-08", "Выступление перед командой", "Soft Skills", "Communication", 1, 5),
        ("EV-09", "Менторство нового коллеги", "Soft Skills", "Leadership", 1, 4),
        ("EV-10", "Управление командой: симуляция", "Soft Skills", "Leadership", 1.5, 5),
        ("EV-11", "A/B-тесты и продуктовые гипотезы", "Hard Skills", "Analytics", 1, 5),
        ("EV-12", "Аналитический проект с наставником", "Hard Skills", "Analytics", 1.5, 4),
        ("EV-13", "Архитектурное ревью с наставником", "Hard Skills", "System Design", 1, 5),
    ]
    events = [dict(event_id=eid, title=title, type=kind, skill=skill, gain=gain, max_level=limit)
              for eid, title, kind, skill, gain, limit in activities]
    history_rows = [
        ("EMP-001", "Soft Skills", "SKIPPED"), ("EMP-001", "Soft Skills", "REJECTED"),
        ("EMP-001", "Soft Skills", "SKIPPED"), ("EMP-001", "Hard Skills", "COMPLETED"),
        ("EMP-002", "Hard Skills", "COMPLETED"), ("EMP-002", "Soft Skills", "SKIPPED"),
        ("EMP-003", "Hard Skills", "COMPLETED"), ("EMP-003", "Soft Skills", "COMPLETED"),
        ("EMP-004", "Soft Skills", "COMPLETED"), ("EMP-004", "Hard Skills", "COMPLETED"),
        ("EMP-005", "Hard Skills", "SKIPPED"), ("EMP-006", "Hard Skills", "COMPLETED"),
    ]
    history = pd.DataFrame(history_rows, columns=HISTORY_COLUMNS[:3])
    history["event_id"], history["completed_at"] = "", ""
    skills = {"grades": {}, "roles": {
        "Backend Developer": {
            "Middle": {"Python": 3, "System Design": 2, "SQL": 3, "Communication": 2},
            "Senior": {"Python": 4, "System Design": 4, "SQL": 4, "Communication": 3, "Leadership": 2},
            "Lead": {"Python": 4, "System Design": 5, "SQL": 4, "Communication": 4, "Leadership": 5},
        },
        "Data Analyst": {
            "Middle": {"Python": 3, "SQL": 3, "Analytics": 3, "Communication": 2},
            "Senior": {"Python": 4, "SQL": 5, "Analytics": 4, "Communication": 4, "Leadership": 2},
            "Lead": {"Python": 4, "SQL": 5, "Analytics": 5, "Communication": 5, "Leadership": 4},
        },
    }}
    return dict(employees=employees, events=events, activity_history=history, skills=skills)


def clean_text(value: Any, default: str = "") -> str:
    if value is None or isinstance(value, (dict, list, bool)):
        return default
    if isinstance(value, float) and not math.isfinite(value):
        return default
    return str(value).strip() or default


def finite_number(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("логическое значение не является уровнем")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("число должно быть конечным")
    return number


def bounded_number(value: Any, default: float, low: float, high: float,
                   issues: list[str], label: str) -> float:
    try:
        number = finite_number(value)
    except (TypeError, ValueError, OverflowError):
        issues.append(f"{label}: некорректное число, использовано {default:g}.")
        return default
    bounded = min(high, max(low, number))
    if bounded != number:
        issues.append(f"{label}: значение ограничено диапазоном {low:g}–{high:g}.")
    return bounded


def normalize_skill_map(raw: Any, issues: list[str], label: str,
                        requirements: bool = False) -> dict[str, float]:
    if not isinstance(raw, dict):
        issues.append(f"{label}: ожидался словарь навыков; использован пустой словарь.")
        return {}
    result = {}
    for skill, value in raw.items():
        name = clean_text(skill)
        try:
            if not name:
                raise ValueError("пустое название")
            number = finite_number(value)
            result[name] = min(5.0, max(0.0, number))
            if number != result[name]:
                issues.append(f"{label}/{name}: уровень ограничен диапазоном 0–5.")
        except (TypeError, ValueError, OverflowError):
            if name and not requirements:
                result[name] = 0.0
            action = "требование пропущено" if requirements else "уровень принят за 0"
            issues.append(f"{label}/{name}: некорректный уровень, {action}.")
    return result


def records_from_json(raw: Any, key: str, id_field: str) -> list[Any]:
    if isinstance(raw, dict) and key in raw:
        raw = raw[key]
    if isinstance(raw, dict) and id_field in raw:
        return [raw]
    if isinstance(raw, dict):
        return [dict(value, **{id_field: value.get(id_field, record_id)})
                if isinstance(value, dict) else value for record_id, value in raw.items()]
    if not isinstance(raw, list):
        raise ValueError(f"Ожидался массив записей или объект с ключом «{key}».")
    return raw


def normalize_employees(raw: Any, issues: list[str]) -> list[dict[str, Any]]:
    rows = records_from_json(raw, "employees", "employee_id")
    result, seen = [], set()
    for index, row in enumerate(rows, 1):
        label = f"Сотрудник, запись {index}"
        try:
            employee_id = clean_text(row["employee_id"])
            if not employee_id or employee_id in seen:
                raise ValueError("пустой или повторяющийся employee_id")
            missing = [key for key in ("role", "grade", "next_grade", "skills", "tenure_months")
                       if key not in row]
            if missing:
                issues.append(f"{label}: нет {', '.join(missing)}; применены значения по умолчанию.")
            grade = clean_text(row.get("grade"), "Не указан")
            result.append(dict(
                employee_id=employee_id, role=clean_text(row.get("role"), "Не указана"), grade=grade,
                next_grade=clean_text(row.get("next_grade"), NEXT_GRADES.get(grade.casefold(), "Не указан")),
                skills=normalize_skill_map(row.get("skills", {}), issues, label),
                tenure_months=int(bounded_number(row.get("tenure_months", 0), 0, 0, 1200, issues, label)),
            ))
            seen.add(employee_id)
        except (KeyError, TypeError, AttributeError, ValueError) as error:
            issues.append(f"{label}: запись пропущена ({error}).")
    if rows and not result:
        raise ValueError("Нет пригодных профилей. Укажите уникальный employee_id для каждого сотрудника.")
    return result


def normalize_events(raw: Any, issues: list[str]) -> list[dict[str, Any]]:
    rows = records_from_json(raw, "events", "event_id")
    result, seen = [], set()
    for index, row in enumerate(rows, 1):
        label = f"Активность, запись {index}"
        try:
            event_id, skill = clean_text(row["event_id"]), clean_text(row["skill"])
            if not event_id or not skill or event_id in seen:
                raise ValueError("пустые event_id/skill или повторяющийся event_id")
            missing = [key for key in ("title", "type", "gain", "max_level") if key not in row]
            if missing:
                issues.append(f"{label}: нет {', '.join(missing)}; применены значения по умолчанию.")
            result.append(dict(
                event_id=event_id, skill=skill, title=clean_text(row.get("title"), event_id),
                type=clean_text(row.get("type"), "Other"),
                gain=bounded_number(row.get("gain", 1), 1, 0, 5, issues, f"{label}/gain"),
                max_level=bounded_number(row.get("max_level", 5), 5, 0, 5, issues, f"{label}/max_level"),
            ))
            seen.add(event_id)
        except (KeyError, TypeError, AttributeError, ValueError) as error:
            issues.append(f"{label}: запись пропущена ({error}).")
    if rows and not result:
        raise ValueError("Нет пригодных активностей. Обязательны уникальный event_id и непустой skill.")
    return result


def normalize_requirements(raw: Any, issues: list[str]) -> dict[str, Any]:
    """Accept global grades, role/grade matrices, or rows with grade and skills."""
    result: dict[str, Any] = {"grades": {}, "roles": {}}
    if isinstance(raw, dict) and set(raw) == {"skills"}:
        raw = raw["skills"]

    def add_grade(role: str, grade: Any, levels: Any) -> None:
        grade_name = clean_text(grade)
        if not grade_name:
            issues.append("Требование без grade пропущено.")
            return
        if isinstance(levels, dict) and "skills" in levels:
            levels = levels["skills"]
        normalized = normalize_skill_map(levels, issues, f"{role or 'Общие'}/{grade_name}", True)
        if normalized:
            destination = result["roles"].setdefault(role, {}) if role else result["grades"]
            destination[grade_name] = normalized

    if isinstance(raw, list):
        for index, row in enumerate(raw, 1):
            try:
                add_grade(clean_text(row.get("role")), row["grade"], row["skills"])
            except (KeyError, AttributeError, TypeError) as error:
                issues.append(f"Требования, запись {index}: пропущена ({error}).")
    elif isinstance(raw, dict):
        if "grades" in raw or "roles" in raw:
            global_grades, roles = raw.get("grades", {}), raw.get("roles", {})
        else:
            global_grades, roles = {}, {}
            for key, value in raw.items():
                if isinstance(value, dict) and ("skills" in value or
                        not any(isinstance(item, dict) for item in value.values())):
                    global_grades[key] = value
                else:
                    roles[key] = value
        if not isinstance(global_grades, dict) or not isinstance(roles, dict):
            raise ValueError("Поля grades и roles должны быть объектами JSON.")
        for grade, levels in global_grades.items():
            add_grade("", grade, levels)
        for role, grades in roles.items():
            if not isinstance(grades, dict):
                issues.append(f"Роль {role}: ожидался словарь грейдов, запись пропущена.")
                continue
            for grade, levels in grades.items():
                add_grade(clean_text(role), grade, levels)
    else:
        raise ValueError("skills.json должен содержать объект требований или массив записей.")
    if not result["grades"] and not result["roles"]:
        explicit_empty = raw == {} or raw == [] or raw == {"grades": {}, "roles": {}}
        if not explicit_empty:
            raise ValueError("Не найдено пригодных требований: нужен словарь «грейд → навык → уровень 0–5».")
    return result


def normalize_history(raw: bytes, issues: list[str]) -> pd.DataFrame:
    try:
        frame = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        issues.append("CSV пуст: загружена пустая история.")
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("CSV содержит повторяющиеся названия колонок.")
    missing = [column for column in HISTORY_COLUMNS[:3] if column not in frame.columns]
    if missing:
        raise ValueError(f"В CSV отсутствуют обязательные колонки: {', '.join(missing)}.")
    for column in HISTORY_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].map(clean_text)
    frame["status"] = frame["status"].str.upper()
    valid = frame["status"].isin(STATUSES) & frame["employee_id"].ne("") & frame["event_type"].ne("")
    invalid_count = int((~valid).sum())
    if invalid_count:
        issues.append(f"История: пропущено некорректных строк — {invalid_count}.")
    if len(frame) and not valid.any():
        raise ValueError("В истории нет корректных строк с employee_id, event_type и известным status.")
    return frame.loc[valid, HISTORY_COLUMNS].reset_index(drop=True)


def parse_upload(dataset: str, raw: bytes) -> tuple[Any, list[str]]:
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("Максимальный размер одного файла — 10 МБ.")
    issues: list[str] = []
    if dataset == "activity_history":
        return normalize_history(raw, issues), issues
    payload = json.loads(raw.decode("utf-8-sig"))
    parser = {"employees": normalize_employees, "events": normalize_events, "skills": normalize_requirements}[dataset]
    return parser(payload, issues), issues


def initialize_state() -> None:
    for key, value in mock_data().items():
        if key not in st.session_state:
            st.session_state[key] = value
    for key, value in {"upload_hashes": {}, "upload_reports": {}, "sources": {}, "upload_epoch": 0}.items():
        if key not in st.session_state:
            st.session_state[key] = value


def lookup_casefold(mapping: dict[str, Any], key: str) -> Any:
    return next((value for name, value in mapping.items() if name.casefold() == key.casefold()), None)


def grade_requirements(employee: dict[str, Any]) -> dict[str, float]:
    matrix = st.session_state.skills
    general = lookup_casefold(matrix["grades"], employee["next_grade"]) or {}
    role_matrix = lookup_casefold(matrix["roles"], employee["role"]) or {}
    specific = lookup_casefold(role_matrix, employee["next_grade"]) or {}
    return {**general, **specific}


def canonical_type(value: str) -> str:
    normalized = " ".join(value.casefold().replace("-", " ").replace("_", " ").split())
    if normalized in {"soft", "soft skill", "soft skills", "софт", "софт скиллы", "гибкие навыки"}:
        return "soft skills"
    if normalized in {"hard", "hard skill", "hard skills", "хард", "хард скиллы", "технические навыки"}:
        return "hard skills"
    return normalized


def employee_history(employee_id: str) -> pd.DataFrame:
    history = st.session_state.activity_history
    return history.loc[history["employee_id"].eq(employee_id)]


def effective_gain(employee: dict[str, Any], event: dict[str, Any]) -> float:
    current = employee["skills"].get(event["skill"], 0.0)
    return max(0.0, min(event["gain"], min(5.0, event["max_level"]) - current))


def score_candidates(employee: dict[str, Any]) -> list[dict[str, Any]]:
    requirements = grade_requirements(employee)
    history = employee_history(employee["employee_id"])
    skipped = history.loc[history["status"].isin(["SKIPPED", "REJECTED"]), "event_type"]
    skip_counts = Counter(canonical_type(value) for value in skipped)
    completed = set(history.loc[history["status"].eq("COMPLETED"), "event_id"]) - {""}
    candidates = []
    for event in st.session_state.events:
        skill = event["skill"]
        current = employee["skills"].get(skill, 0.0)
        required = requirements.get(skill, 0.0)
        gap = max(0.0, required - current)
        gain = effective_gain(employee, event)
        if gain <= 0 or event["event_id"] in completed or (requirements and gap <= 0):
            continue
        kind = canonical_type(event["type"])
        skip_count = skip_counts[kind]
        penalty = 5 * skip_count if skip_count >= 2 else 0
        useful_gain = min(gain, gap) if requirements else gain
        importance = required / 5.0 if requirements else 0.0
        hard_bonus = 2.0 if kind == "hard skills" and gap > 0 and skip_counts["soft skills"] >= 2 else 0.0
        score = 10 * importance + 3 * gap - penalty + 2 * useful_gain + hard_bonus
        if requirements:
            requirement_text = (
                f"{skill}: {current:g} из требуемых {required:g} для {employee['next_grade']}. "
                f"Важность для грейда: {importance:.2f} × 10 = {10 * importance:g} балла. "
                f"Разрыв {gap:g} × 3 = {3 * gap:g} балла. "
                f"Активность закрывает {useful_gain:g} из {gap:g} пункта разрыва. "
            )
        else:
            requirement_text = (
                f"{skill}: текущий уровень {current:g}. Требования для {employee['next_grade']} "
                "не загружены: важность для грейда и разрыв не оцениваются (0 баллов). "
                "Это общее развитие навыка; связь с переходом грейда не подтверждена. "
            )
        if penalty:
            history_text = (f"История типа «{event['type']}»: пропусков/отказов {skip_count}; "
                            f"штраф −5 × {skip_count} = −{penalty} баллов. ")
        else:
            history_text = (f"История типа «{event['type']}»: пропусков/отказов {skip_count}; "
                            "штраф 0 (применяется от двух). ")
        bonus_text = (
            f"Soft Skills пропущены/отклонены {skip_counts['soft skills']} раза: "
            "приоритетный Hard Skill с разрывом получает ещё +2 балла. " if hard_bonus else ""
        )
        explanation = (
            requirement_text + history_text + bonus_text +
            f"Заявленный прирост +{event['gain']:g}, лимит {event['max_level']:g}/5: "
            f"реальный прирост +{gain:g}, уровень после выполнения {current + gain:g}/5. "
            f"Полезный прирост × 2 = {2 * useful_gain:g} балла. Итого: {score:g}."
        )
        candidates.append(dict(event=event, score=score, current=current, required=required,
                               gap=gap, effective_gain=gain, useful_gain=useful_gain,
                               grade_points=10 * importance, gap_points=3 * gap,
                               penalty=penalty, hard_bonus=hard_bonus, explanation=explanation))
    return sorted(candidates, key=lambda item: (-item["score"], -item["useful_gain"], item["event"]["event_id"]))


def calculate_recommendations(employee: dict[str, Any]) -> list[dict[str, Any]]:
    """Return up to three distinct skills, taking the best-scoring event for each."""
    recommendations, seen_skills = [], set()
    for candidate in score_candidates(employee):
        skill = candidate["event"]["skill"]
        if skill in seen_skills:
            continue
        recommendations.append(candidate)
        seen_skills.add(skill)
        if len(recommendations) == 3:
            break
    return recommendations


def complete_activity(employee_id: str, event_id: str) -> tuple[bool, str]:
    try:
        employee = next(item for item in st.session_state.employees if item["employee_id"] == employee_id)
        event = next(item for item in st.session_state.events if item["event_id"] == event_id)
    except StopIteration:
        return False, "Сотрудник или активность больше не найдены в загруженных данных."
    history = employee_history(employee_id)
    if ((history["event_id"] == event_id) & (history["status"] == "COMPLETED")).any():
        return False, "Эта активность уже выполнена; повторное начисление не требуется."
    gain = effective_gain(employee, event)
    if gain <= 0:
        return False, "Лимит активности уже достигнут; уровень навыка сохранён."
    old_level = employee["skills"].get(event["skill"], 0.0)
    new_level = min(5.0, event["max_level"], round(old_level + gain, 10))
    row = dict(employee_id=employee_id, event_type=event["type"], status="COMPLETED",
               event_id=event_id, completed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    updated_history = pd.concat([st.session_state.activity_history, pd.DataFrame([row])], ignore_index=True)
    employee["skills"][event["skill"]] = new_level
    st.session_state.activity_history = updated_history
    return True, (f"{event['title']} выполнена. {event['skill']}: {old_level:g} → {new_level:g}. "
                  "Рекомендации пересчитаны.")


def readiness(employee: dict[str, Any]) -> float | None:
    requirements = grade_requirements(employee)
    total = sum(requirements.values())
    if total <= 0:
        return None
    covered = sum(min(employee["skills"].get(skill, 0), level) for skill, level in requirements.items())
    return 100 * covered / total


def company_gaps() -> pd.DataFrame:
    rows = []
    for employee in st.session_state.employees:
        for skill, required in grade_requirements(employee).items():
            if required <= 0:
                continue
            current = employee["skills"].get(skill, 0.0)
            rows.append({"employee_id": employee["employee_id"], "Должность": employee["role"],
                         "Грейд": employee["grade"], "Цель": employee["next_grade"],
                         "Навык": skill, "Текущий": current, "Требуется": required,
                         "Разрыв": max(0.0, required - current)})
    return pd.DataFrame(rows, columns=["employee_id", "Должность", "Грейд", "Цель",
                                       "Навык", "Текущий", "Требуется", "Разрыв"])


def company_history() -> pd.DataFrame:
    ids = {employee["employee_id"] for employee in st.session_state.employees}
    history = st.session_state.activity_history
    return history.loc[history["employee_id"].isin(ids)]


def engagement(history: pd.DataFrame) -> tuple[float | None, int, int]:
    completed = int(history["status"].eq("COMPLETED").sum())
    missed = int(history["status"].isin(["SKIPPED", "REJECTED"]).sum())
    total = completed + missed
    return (100 * completed / total if total else None), completed, missed


def data_archive(data: dict[str, Any]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for dataset in ("employees", "events", "skills"):
            archive.writestr(f"{dataset}.json", json.dumps(data[dataset], ensure_ascii=False, indent=2))
        archive.writestr("activity_history.csv", data["activity_history"].to_csv(index=False).encode("utf-8-sig"))
    return buffer.getvalue()


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## ◈ Career Quest")
        st.caption("Следующий шаг в вашей карьере")
        mode = st.radio("Режим", ["Сотрудник", "HR-Дашборд"], key="mode")
        st.divider()
        st.markdown("### Загрузка данных Жюри")
        st.caption("UTF-8 · до 10 МБ на файл. Каждый файл заменяет свой набор данных.")
        for dataset, filename, extension in [
            ("employees", "employees.json", "json"), ("events", "events.json", "json"),
            ("activity_history", "activity_history.csv", "csv"), ("skills", "skills.json", "json"),
        ]:
            uploaded = st.file_uploader(filename, type=[extension],
                                        key=f"upload_{dataset}_{st.session_state.upload_epoch}")
            if uploaded is None:
                st.session_state.upload_hashes.pop(dataset, None)
            else:
                raw = uploaded.getvalue()
                fingerprint = hashlib.sha256(raw).hexdigest()
                if st.session_state.upload_hashes.get(dataset) != fingerprint:
                    try:
                        parsed, issues = parse_upload(dataset, raw)
                        st.session_state[dataset] = parsed
                        st.session_state.sources[dataset] = uploaded.name
                        st.session_state.upload_reports[dataset] = (True, issues)
                    except (ValueError, TypeError, KeyError, AttributeError, UnicodeError,
                            pd.errors.ParserError, OverflowError, RecursionError) as error:
                        st.session_state.upload_reports[dataset] = (False, [str(error)])
                    st.session_state.upload_hashes[dataset] = fingerprint
            st.caption(f"Источник: {st.session_state.sources.get(dataset, 'мок-данные')}")
            report = st.session_state.upload_reports.get(dataset)
            if report:
                success, issues = report
                if success:
                    st.success("Данные загружены" if not issues else "Загружено с предупреждениями")
                    if issues:
                        with st.expander(f"Исправления и пропуски ({filename}): {len(issues)}"):
                            for issue in issues[:30]:
                                st.write(issue)
                            if len(issues) > 30:
                                st.caption(f"Ещё предупреждений: {len(issues) - 30}")
                else:
                    st.error(f"Файл не применён; предыдущие данные сохранены. {issues[0]}")
        with st.expander("Форматы и правила загрузки"):
            st.write("employees.json и events.json: массив объектов; также поддерживаются оболочки employees/events и словари по ID.")
            st.code('[{"employee_id":"E1","role":"Developer","grade":"Middle",\n'
                    '  "next_grade":"Senior","skills":{"Python":2},"tenure_months":18}]', language="json")
            st.code('[{"event_id":"A1","title":"Python Pro","type":"Hard Skills",\n'
                    '  "skill":"Python","gain":1,"max_level":4}]', language="json")
            st.code('employee_id,event_type,status\nE1,Hard Skills,COMPLETED\nE1,Soft Skills,SKIPPED', language="csv")
            st.write("skills.json: общие требования по грейдам или требования с учётом должности.")
            st.code('{"Senior":{"Python":4,"Communication":3}}', language="json")
            st.code('{"roles":{"Developer":{"Senior":{"Python":4}}},"grades":{}}', language="json")
            st.write("Также принимается массив записей {role, grade, skills}. Названия навыков должны совпадать во всех файлах. Роль и грейд сопоставляются без учёта регистра.")
            st.write("Отсутствующий навык сотрудника считается равным 0. Для активности значения по умолчанию: gain=1, max_level=5, type=Other. Записи без ID или навыка активности пропускаются. Ошибка обязательных колонок CSV сохраняет прежнюю историю.")
            st.write("event_id и completed_at в CSV необязательны. При выполнении event_id записывается для защиты от повторного начисления. История без event_id влияет на скоринг, но не определяет конкретную завершённую активность.")
        st.download_button("Скачать примеры всех 4 файлов", data=data_archive(mock_data()),
                           file_name="career_quest_examples.zip", mime="application/zip", width="stretch")
        st.download_button("Скачать текущие данные", data=data_archive({key: st.session_state[key]
                           for key in ("employees", "events", "skills", "activity_history")}),
                           file_name="career_quest_progress.zip", mime="application/zip", width="stretch")
        if st.button("Восстановить мок-данные", width="stretch"):
            for key, value in mock_data().items():
                st.session_state[key] = value
            st.session_state.upload_hashes = {}
            st.session_state.upload_reports = {}
            st.session_state.sources = {}
            st.session_state.upload_epoch += 1
            st.session_state.pop("selected_employee", None)
            st.session_state.pop("flash", None)
            st.rerun()
        st.caption("Данные и прогресс хранятся только в текущей сессии браузера. Скачайте их перед закрытием.")
    return mode


def render_scoring_notes() -> None:
    with st.expander("Как AI-скоринг выбирает следующий шаг"):
        st.write("Объяснимый рекомендательный алгоритм, без внешней модели и API.")
        st.code("score = 10 × (требуемый уровень / 5)\n"
                "      + 3 × max(требуемый − текущий, 0)\n"
                "      − 5 × число пропусков/отказов типа (если их ≥ 2)\n"
                "      + 2 × полезный прирост\n"
                "      + 2 за Hard Skills с разрывом при ≥ 2 пропусках Soft Skills", language="text")
        st.write("Реальный прирост = max(0, min(gain, max_level − текущий, 5 − текущий)). "
                 "Полезный прирост ограничен оставшимся разрывом. Активности без прироста, "
                 "выполненные активности и уже закрытые требования исключаются. "
                 "Показываются до трёх разных навыков с лучшими баллами; отрицательный балл не запрещает рекомендацию.")
        st.caption("Без требований грейда доступны рекомендации общего развития по приросту и истории; готовность к грейду не рассчитывается. Стаж отображается в профиле и не влияет на скоринг.")


def render_employee() -> None:
    st.title("Ваш следующий карьерный шаг")
    st.caption("Понятная цель. Персональный маршрут. Видимый прогресс.")
    employees = st.session_state.employees
    if not employees:
        st.info("Список сотрудников пуст. Загрузите employees.json или восстановите мок-данные.")
        return
    ids = [employee["employee_id"] for employee in employees]
    if st.session_state.get("selected_employee") not in ids:
        st.session_state.selected_employee = ids[0]
    selected = st.selectbox("Выберите employee_id", ids, key="selected_employee")
    employee = next(item for item in employees if item["employee_id"] == selected)
    requirements = grade_requirements(employee)
    flash = st.session_state.pop("flash", None)
    if flash:
        (st.success if flash[0] else st.warning)(flash[1])
    with st.container(border=True):
        st.subheader(employee["role"])
        columns = st.columns(4)
        columns[0].metric("Сотрудник", employee["employee_id"])
        columns[1].metric("Текущий грейд", employee["grade"])
        columns[2].metric("Целевой грейд", employee["next_grade"])
        columns[3].metric("Стаж, месяцев", employee["tenure_months"])
        percent = readiness(employee)
        if percent is not None:
            st.progress(min(1.0, max(0.0, percent / 100)),
                        text=f"Покрытие требований {employee['next_grade']}: {percent:.0f}%")
            st.caption("Покрытие навыков показывает прогресс развития; решение о повышении принимает компания.")
        else:
            st.warning("Нет положительных требований для этой должности и целевого грейда. Загрузите skills.json для оценки готовности.")
    left, right = st.columns([1, 1.65], gap="large")
    with left:
        st.subheader("Карта навыков")
        all_skills = sorted(set(employee["skills"]) | set(requirements),
                            key=lambda skill: (-max(0, requirements.get(skill, 0) - employee["skills"].get(skill, 0)), skill))
        if not all_skills:
            st.info("Навыки пока не указаны.")
        for skill in all_skills:
            current = employee["skills"].get(skill, 0.0)
            st.text(f"{skill} · {current:g}/5")
            st.progress(min(1.0, max(0.0, current / 5)))
            if skill in requirements:
                gap = max(0.0, requirements[skill] - current)
                st.caption(f"Цель {requirements[skill]:g}/5 · " + (f"разрыв {gap:g}" if gap else "требование выполнено"))
        render_scoring_notes()
    with right:
        st.subheader("Рекомендованные шаги развития")
        recommendations = calculate_recommendations(employee)
        gaps = {skill for skill, required in requirements.items() if required > employee["skills"].get(skill, 0)}
        eligible_skills = {item["event"]["skill"] for item in score_candidates(employee)}
        uncovered = gaps - eligible_skills
        if uncovered:
            st.warning("В каталоге нет доступных шагов для разрывов: " + ", ".join(sorted(uncovered)) +
                       ". Добавьте активности с подходящим лимитом.")
        if not recommendations:
            if requirements and not gaps:
                st.success("Все заданные требования целевого грейда закрыты. Обсудите следующий этап с руководителем.")
            else:
                st.info("Подходящих активностей пока нет. Обновите каталог events.json или требования skills.json.")
        for index, recommendation in enumerate(recommendations, 1):
            event = recommendation["event"]
            with st.container(border=True):
                st.caption(f"ШАГ {index:02d} · {event['type']} · {recommendation['score']:g} балла")
                st.subheader(event["title"])
                st.text(f"{event['skill']} · +{recommendation['effective_gain']:g} · лимит {event['max_level']:g}/5")
                st.info(recommendation["explanation"], icon="💡")
                if st.button("Выполнить активность", key=f"complete_{selected}_{event['event_id']}",
                             type="primary", width="stretch"):
                    st.session_state.flash = complete_activity(selected, event["event_id"])
                    st.rerun()
        with st.expander("Моя история активностей"):
            history = employee_history(selected)
            if history.empty:
                st.caption("История пока пуста.")
            else:
                st.dataframe(history.iloc[::-1], hide_index=True, width="stretch")


def render_hr() -> None:
    st.title("Развитие команды в цифрах")
    st.caption("Дефицит компетенций относительно следующего грейда каждого сотрудника.")
    employees = st.session_state.employees
    gaps = company_gaps()
    history = company_history()
    rate, completed, missed = engagement(history)
    percentages = [value for employee in employees if (value := readiness(employee)) is not None]
    metrics = st.columns(4)
    metrics[0].metric("Сотрудников", len(employees))
    metrics[1].metric("Среднее покрытие цели", f"{sum(percentages) / len(percentages):.0f}%" if percentages else "Нет данных")
    metrics[2].metric("Вовлечённость", f"{rate:.1f}%" if rate is not None else "Нет истории")
    metrics[3].metric("Сотрудников с разрывом", int(gaps.loc[gaps["Разрыв"] > 0, "employee_id"].nunique()) if not gaps.empty else 0)
    st.caption(f"Вовлечённость = COMPLETED / (COMPLETED + SKIPPED + REJECTED) × 100%. "
               f"Выполнено: {completed}; пропущено/отклонено: {missed}. "
               f"Покрытие рассчитано для {len(percentages)} из {len(employees)} сотрудников.")
    unmatched = len(st.session_state.activity_history) - len(history)
    if unmatched:
        st.info(f"Строк истории с employee_id вне текущего списка: {unmatched}. Они сохранены, но не включены в метрики компании.")
    if not employees:
        st.info("Загрузите сотрудников для просмотра аналитики компании.")
        return
    if len(percentages) < len(employees):
        st.warning("Для части сотрудников нет требований следующего грейда; они не включены в оценку покрытия и дефицита навыков.")
    st.subheader("Какие компетенции проседают сильнее")
    if gaps.empty:
        st.info("Нет требований для расчёта дефицита. Загрузите skills.json с подходящими грейдами и ролями.")
    else:
        aggregate = gaps.groupby("Навык").agg(
            **{"Средний разрыв": ("Разрыв", "mean"), "Суммарный разрыв": ("Разрыв", "sum"),
               "Сотрудников с дефицитом": ("Разрыв", lambda values: int((values > 0).sum()))}
        ).sort_values("Средний разрыв", ascending=False)
        deficits = aggregate.loc[aggregate["Суммарный разрыв"] > 0]
        if deficits.empty:
            st.success("Заданные требования навыков выполнены во всей компании.")
        else:
            st.bar_chart(deficits[["Средний разрыв"]], color="#7367F0", width="stretch")
            st.caption("Средний разрыв рассчитан среди сотрудников, для которых навык требуется, включая сотрудников с нулевым разрывом.")
            st.dataframe(deficits.round(2), width="stretch")
    st.subheader("Сотрудники с дефицитом ключевых навыков")
    st.caption("Ключевые навыки — все навыки с положительным требованием целевого грейда. Каждая строка — один разрыв сотрудника.")
    if not gaps.empty:
        threshold = st.slider("Минимальный разрыв для отображения", 0.0, 5.0, 0.5, 0.5)
        low = gaps.loc[(gaps["Разрыв"] > 0) & (gaps["Разрыв"] >= threshold)].sort_values(
            ["Разрыв", "employee_id"], ascending=[False, True])
        if low.empty:
            st.info("Нет сотрудников с разрывом выше выбранного порога.")
        else:
            st.dataframe(low, hide_index=True, width="stretch")
            st.download_button("Скачать таблицу дефицита CSV", low.to_csv(index=False).encode("utf-8-sig"),
                               file_name="career_quest_skill_gaps.csv", mime="text/csv")
    st.subheader("Участие в активностях")
    if history.empty:
        st.info("Нет истории для текущих сотрудников. Выполните активность в режиме сотрудника или загрузите CSV.")
    else:
        participation = pd.crosstab(history["event_type"], history["status"]).reindex(
            columns=["COMPLETED", "SKIPPED", "REJECTED"], fill_value=0)
        participation["Вовлечённость, %"] = (100 * participation["COMPLETED"] / participation.sum(axis=1)).round(1)
        participation.index.name = "Тип активности"
        st.dataframe(participation, width="stretch")


def main() -> None:
    st.set_page_config(page_title="Career Quest · Развитие сотрудников", page_icon="◈", layout="wide")
    st.markdown("""<style>
        .block-container {padding-top: 2.5rem; padding-bottom: 3rem; max-width: 1440px;}
        [data-testid="stSidebar"] {border-right: 1px solid rgba(115,103,240,.18);}
        [data-testid="stMetric"] {border-radius: 12px; padding: 14px; background: rgba(115,103,240,.075);}
        [data-testid="stMetricValue"] {font-size: 1.7rem;}
        [data-testid="stProgressBar"] > div > div > div > div {background-color: #7367F0;}
        h1 {letter-spacing: -.035em;} h2, h3 {letter-spacing: -.02em;}
        .stButton > button[kind="primary"] {background-color: #6556DB; border-color: #6556DB;}
        </style>""", unsafe_allow_html=True)
    initialize_state()
    mode = render_sidebar()
    if mode == "Сотрудник":
        render_employee()
    else:
        render_hr()


if __name__ == "__main__":
    main()
