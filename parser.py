import os
import re
import json
import zipfile
import requests
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict
from pathlib import Path
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import openai

# Импорт конфигурации МЕДКОМ
try:
    import config
    CONFIG_LOADED = True
except ImportError:
    CONFIG_LOADED = False
    print("[!] config.py не найден - работаем без умного поиска")

load_dotenv()

logger = logging.getLogger("parser")

# ============================================================
# AI-АНАЛИЗ ДОКУМЕНТАЦИИ ТЕНДЕРА
# ============================================================

class DocumentAnalyzer:
    """
    AI-анализатор документации тендера для оценки релевантности.
    Скачивает документы, читает их содержимое и отправляет в GPT для анализа.
    """
    
    def __init__(self, openai_key: str = None):
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY")
        
        # Lazy import DocumentReader
        try:
            from document_reader import DocumentReader
            self.reader = DocumentReader()
        except ImportError:
            self.reader = None
        
        # Загружаем конфиг компании
        try:
            from config import PRODUCTS, COMPANY
            self.products = PRODUCTS
            self.company = COMPANY
        except ImportError:
            self.products = []
            self.company = {"name": "Компания", "description": ""}
    
    def _get_products_description(self) -> str:
        """Формирует описание продукции компании для промпта"""
        if not self.products:
            return "Продукция не указана"
        
        desc = []
        for p in self.products:
            product_desc = f"• {p.get('name', 'Продукт')}:"
            if p.get('keywords'):
                product_desc += f"\n  Ключевые слова: {', '.join(p['keywords'][:10])}"
            if p.get('context_pairs'):
                pairs = [f"{w1}+{w2}" for w1, w2 in p['context_pairs'][:5]]
                product_desc += f"\n  Контекстные пары: {', '.join(pairs)}"
            desc.append(product_desc)
        
        return "\n".join(desc)
    
    async def analyze_tender_documents(
        self, 
        tender_url: str, 
        tender_title: str,
        tender_number: str,
        parser: 'RostenderParser'
    ) -> dict:
        """
        Анализирует документацию тендера и возвращает % релевантности.
        
        Returns:
            {
                "relevance": int (0-100),
                "ai_comment": str,
                "matched_products": list,
                "red_flags": list,
                "analyzed_docs_count": int
            }
        """
        import httpx
        
        result = {
            "relevance": 0,
            "ai_comment": "",
            "matched_products": [],
            "red_flags": [],
            "analyzed_docs_count": 0
        }
        
        if not self.openai_key:
            result["ai_comment"] = "API ключ не настроен"
            return result
        
        try:
            # 1. Получаем детали тендера с документами
            details = parser.get_details(tender_url)
            documents = details.get("documents", [])
            
            if not documents:
                # Анализируем только заголовок и описание
                text_to_analyze = f"Название: {tender_title}\n\nОписание: {details.get('description', '')}"
            else:
                # 2. Скачиваем документы (максимум 3 для скорости)
                docs_to_download = documents[:3]
                files_dir = parser.download_files(docs_to_download, f"analysis_{tender_number}")
                
                if files_dir and self.reader:
                    # 3. Читаем содержимое документов
                    doc_content = self.reader.read_directory(files_dir, recursive=True)
                    result["analyzed_docs_count"] = len(docs_to_download)
                    
                    # Ограничиваем размер текста (макс 15000 символов для GPT)
                    if len(doc_content) > 15000:
                        doc_content = doc_content[:15000] + "\n\n[... текст обрезан ...]"
                    
                    text_to_analyze = f"Название тендера: {tender_title}\n\nОписание: {details.get('description', '')}\n\n=== СОДЕРЖИМОЕ ДОКУМЕНТАЦИИ ===\n{doc_content}"
                else:
                    text_to_analyze = f"Название: {tender_title}\n\nОписание: {details.get('description', '')}"
            
            # 4. Отправляем в GPT для анализа
            products_desc = self._get_products_description()
            company_name = self.company.get("name", "Компания")
            company_desc = self.company.get("description", "")
            
            prompt = f"""Ты эксперт по госзакупкам и тендерам. Проанализируй документацию тендера и оцени его релевантность для компании-производителя.

=== КОМПАНИЯ ===
Название: {company_name}
Описание: {company_desc}

=== ПРОДУКЦИЯ КОМПАНИИ ===
{products_desc}

=== ДОКУМЕНТАЦИЯ ТЕНДЕРА ===
{text_to_analyze}

=== ЗАДАЧА ===
Оцени, насколько этот тендер подходит для участия компании {company_name}.

ВАЖНЫЕ ПРАВИЛА ОЦЕНКИ:
1. Если в документах явно указаны товары из нашего каталога (контейнеры для медотходов, емкости для биоматериалов, пакеты для утилизации) - высокая релевантность (70-100%)
2. Если контекст медицинский, но товары не наши точно - средняя релевантность (40-69%)
3. Если тендер явно не по нашей тематике (строительство, IT, продукты питания) - низкая релевантность (0-39%)
4. Обрати внимание на НЕГАТИВНЫЙ контекст: "контейнер для мусора", "пакет документов", "строительные емкости" - это НЕ наше!

Ответь СТРОГО в формате JSON:
{{
    "relevance": <число от 0 до 100>,
    "comment": "<краткий комментарий почему такая оценка, 1-2 предложения>",
    "matched_products": ["<названия подходящих продуктов из нашего каталога>"],
    "red_flags": ["<проблемные условия если есть: срочные сроки, специфические требования, единственный поставщик и т.д.>"]
}}

Только JSON, без дополнительного текста!"""

            # Вызываем API
            api_url = os.getenv("OPENAI_BASE_URL", "https://api.proxyapi.ru/openai/v1")
            if not api_url.endswith("/chat/completions"):
                api_url = api_url.rstrip("/") + "/chat/completions"
            
            async with httpx.AsyncClient(timeout=90.0) as client:
                resp = await client.post(
                    api_url,
                    headers={"Authorization": f"Bearer {self.openai_key}"},
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1
                    }
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                
                # Парсим JSON из ответа
                if "```" in content:
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                
                analysis = json.loads(content.strip())
                
                result["relevance"] = int(analysis.get("relevance", 0))
                result["ai_comment"] = analysis.get("comment", "")
                result["matched_products"] = analysis.get("matched_products", [])
                result["red_flags"] = analysis.get("red_flags", [])
                
        except Exception as e:
            logger.error(f"Document analysis error: {e}")
            result["ai_comment"] = f"Ошибка анализа: {str(e)[:50]}"
        
        return result
    
    def analyze_tender_documents_sync(
        self, 
        tender_url: str, 
        tender_title: str,
        tender_number: str,
        parser: 'RostenderParser'
    ) -> dict:
        """Синхронная обёртка для analyze_tender_documents"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.analyze_tender_documents(tender_url, tender_title, tender_number, parser)
        )


# Глобальный экземпляр анализатора
_document_analyzer: Optional[DocumentAnalyzer] = None

def get_document_analyzer() -> DocumentAnalyzer:
    global _document_analyzer
    if _document_analyzer is None:
        _document_analyzer = DocumentAnalyzer()
    return _document_analyzer
if os.getenv("DEBUG", "false").lower() == "true":
    logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)
else:
    logger.addHandler(logging.NullHandler())

@dataclass
class Tender:
    number: str
    title: str
    customer: str
    price: str
    deadline: str
    description: str
    url: str
    region: str = ""  # Регион/место проведения
    relevance: int = 0  # Релевантность 0-100 (из config)
    matched_product: str = ""  # Какой продукт из конфига совпал
    red_flags: List[str] = None  # Найденные красные флаги

    def __post_init__(self):
        if self.red_flags is None:
            self.red_flags = []

@dataclass
class TenderFile:
    id: str
    title: str
    url: str
    extension: str
    local_path: str = ""

class RostenderParser:
    BASE_URL = "https://rostender.info"
    
    DEFAULT_REGIONS = {
        "tatarstan": "20", "kazan": "20", "moscow": "182394", 
        "spb": "155429", "cfo": "1", "pfo": "7"
    }

    def __init__(self, email: str = None, password: str = None, downloads_dir: str = "downloads"):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Upgrade-Insecure-Requests': '1'
        })
        
        self.downloads_dir = Path(downloads_dir)
        self.cache_dir = Path("cache")
        for d in [self.downloads_dir, self.cache_dir]:
            d.mkdir(exist_ok=True)

        self.ai_client = None
        if key := os.getenv('OPENAI_API_KEY'):
            self.ai_client = openai.OpenAI(api_key=key, base_url="https://api.proxyapi.ru/openai/v1")

        self.regions_map = self._load_map('regions')
        self.branches_map = self._load_map('branches')

        self._fetch_remote_data()

        print(f"Загружено {len(self.regions_map)} регионов")
        print(f"Загружено {len(self.branches_map)} отраслей")

        self.is_logged_in = False
        if email and password:
            self._login(email, password)

    def _fetch_remote_data(self):
        try:
            r = self.session.get(f"{self.BASE_URL}/yiiajax/get-regions", headers={'X-Requested-With': 'XMLHttpRequest'})
            if r.status_code == 200:
                data = r.json().get('result', [])

                with open("regions_data.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                for item in data:
                    self.regions_map[item['name'].lower()] = str(item['id'])
                    for d in item.get('districts', []):
                        self.regions_map[d['name'].lower()] = str(d['id'])

                self.regions_map.update({
                    'москва': '182394', 'спб': '155429', 'питер': '155429',
                    'татарстан': '20', 'казань': '20'
                })
                self._save_map('regions', self.regions_map)
        except Exception as e:
            logger.debug(f"Regions fetch error: {e}")

        try:
            r = self.session.get(f"{self.BASE_URL}/yiiajax/get-branches", headers={'X-Requested-With': 'XMLHttpRequest'})
            if r.status_code == 200:
                data = r.json().get('result', [])

                with open("branches_data.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)

                self.branches_map = {}
                for item in data:
                    for child in item.get('children', []):
                        self.branches_map[child['name'].lower()] = str(child['id'])
                        
                self._save_map('branches', self.branches_map)
        except Exception as e:
            logger.debug(f"Branches fetch error: {e}")

    def _login(self, email, password):
        print("Попытка авторизации...")
        try:
            login_url = f"{self.BASE_URL}/login"
            resp = self.session.get(login_url)
            soup = BeautifulSoup(resp.text, 'html.parser')
            csrf = soup.find('input', {'name': re.compile(r'csrf', re.I)})
            
            data = {
                'LoginForm[username]': email,
                'LoginForm[password]': password,
                'LoginForm[rememberMe]': '1'
            }
            if csrf: data['_csrf'] = csrf['value']

            post = self.session.post(login_url, data=data)
            if post.status_code == 200 and 'logout' in post.text.lower():
                self.is_logged_in = True
                print("[OK] Авторизация успешна")
            else:
                print("[X] Авторизация не удалась")
        except Exception as e:
            logger.debug(f"Login error: {e}")
            print("[X] Ошибка авторизации")

    def _load_map(self, name: str) -> dict:
        path = self.cache_dir / f"{name}_map.json"
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f: return json.load(f)
            except: pass
        return {}

    def _save_map(self, name: str, data: dict):
        try:
            with open(self.cache_dir / f"{name}_map.json", 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except: pass

    def _load_json_data(self, filename: str) -> dict:
        if Path(filename).exists():
            with open(filename, 'r', encoding='utf-8') as f: return json.load(f)
        return {}

    def _resolve_entity(self, query: str, entity_type: str, keyword: str = "") -> Optional[str]:
        if not query: return None
        key = query.lower().strip()
        
        cache_map = self.regions_map if entity_type == 'regions' else self.branches_map

        if key in cache_map: 
            return cache_map[key]

        if entity_type == 'regions' and key in self.DEFAULT_REGIONS:
            return self.DEFAULT_REGIONS[key]

        if not self.ai_client: return None

        context = "Список регионов РФ." if entity_type == 'regions' else "Список отраслей."
        additional_context = ""
        if entity_type == 'branches' and keyword:
            additional_context = f" Контекст (ключевое слово): '{keyword}'. Используй его для выбора наиболее подходящей тематики."

        options_list = []
        for name, eid in list(cache_map.items())[:800]:
            options_list.append(f"{eid}: {name}")
        
        options_text = "\n".join(options_list)

        prompt = (
            f"Твоя задача: выбрать из списка ID категории, которая лучше всего подходит под запрос.\n"
            f"Запрос пользователя: '{query}'\n"
            f"{additional_context}\n"
            f"Тип данных: {context}\n\n"
            f"Список доступных вариантов (ID: Название):\n"
            f"{options_text}\n\n"
            f"Правила:\n"
            f"1. Верни СТРОГО и ТОЛЬКО JSON с ID выбранной категории.\n"
            f"2. Формат: {{'id': '123'}}\n"
            f"3. Если ничего не подходит, верни null."
        )

        try:
            resp = self.ai_client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0
            )
            content = resp.choices[0].message.content.replace("```json", "").replace("```", "").strip()
            if content == "null": return None

            res = json.loads(content)
            found_id = str(res.get('id', ''))
            
            if found_id:
                if found_id in cache_map.values():
                    cache_map[key] = found_id
                    self._save_map(entity_type, cache_map)
                    print(f"  [AI] '{query}' -> ID: {found_id}")
                    return found_id
                else:
                    logger.debug(f"AI returned invalid ID: {found_id}")

        except Exception as e:
            logger.debug(f"AI resolution error: {e}")
            pass
        return None

    def search(self, region: str, industry: str, keyword: str, limit: int = 20) -> List[Tender]:
        params = {'keywords': keyword}
        
        rid = self._resolve_entity(region, 'regions')
        if rid: params['geo[]'] = rid

        bid = self._resolve_entity(industry, 'branches', keyword)
        if bid: params['branch[]'] = bid

        r_name = next((k.title() for k, v in self.regions_map.items() if v == rid), rid) if rid else 'Все'
        b_name = next((k.title() for k, v in self.branches_map.items() if v == bid), bid) if bid else 'Все'
        
        print(f"\nПараметры: Регион='{r_name}', Отрасль='{b_name}', Слово='{keyword}'")

        tenders = []
        try:
            resp = self.session.get(f"{self.BASE_URL}/extsearch", params=params, timeout=30)
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            cards = soup.select('article.tender-row') or soup.select('.search-item')
            logger.debug(f"Found {len(cards)} cards")
            
            for card in cards[:limit]:
                if t := self._parse_card(card): tenders.append(t)
        except Exception as e:
            logger.debug(f"Search error: {e}")
            pass
        return tenders

    def _parse_card(self, card) -> Optional[Tender]:
        try:
            # Ссылка на тендер
            link = (card.select_one('a.tender-info__description') or 
                    card.select_one('a.tender-title') or
                    card.select_one('a[href*="/tender"]') or
                    card.select_one('a[href*="/zakupki"]'))
            if not link: return None
            
            href = link['href']
            url = self.BASE_URL + href if href.startswith('/') else href
            
            # Номер тендера
            num = "N/A"
            num_node = (card.select_one('.tender__number') or 
                       card.select_one('.tender-number') or
                       card.select_one('[class*="number"]'))
            if num_node:
                if m := re.search(r'(\d{7,})', num_node.text): 
                    num = m.group(1)

            # Цена
            price = "—"
            p_node = (card.select_one('.starting-price__price') or 
                     card.select_one('.tender-price') or
                     card.select_one('.price') or
                     card.select_one('[class*="price"]'))
            if p_node:
                price = p_node.get_text(strip=True)

            # Дедлайн (несколько вариантов селекторов)
            deadline = ""
            deadline_node = (card.select_one('.tender__countdown-text') or
                           card.select_one('.tender-deadline') or
                           card.select_one('.deadline') or
                           card.select_one('[class*="countdown"]') or
                           card.select_one('[class*="deadline"]') or
                           card.select_one('[class*="date"]'))
            if deadline_node:
                deadline = deadline_node.get_text(strip=True)
            
            # Заказчик
            customer = "Не указан"
            cust_node = (card.select_one('.tender-customer__name') or
                        card.select_one('.tender-customer') or
                        card.select_one('.customer') or
                        card.select_one('[class*="customer"]'))
            if cust_node:
                customer = cust_node.get_text(strip=True)

            # Регион/место
            region = ""
            region_node = (card.select_one('.tender-region') or
                          card.select_one('.tender__region') or
                          card.select_one('.region') or
                          card.select_one('[class*="region"]') or
                          card.select_one('[class*="location"]') or
                          card.select_one('[class*="geo"]'))
            if region_node:
                region = region_node.get_text(strip=True)

            title = link.get_text(strip=True)
            
            logger.debug(f"Parsed: num={num}, title={title[:50]}, deadline={deadline}, region={region}")

            return Tender(
                number=num,
                title=title,
                customer=customer,
                price=price,
                deadline=deadline,
                description=title,
                url=url,
                region=region
            )
        except Exception as e:
            logger.debug(f"Card parse error: {e}")
            return None

    # ============================================================
    # МЕТОДЫ ДЛЯ РАБОТЫ С CONFIG (МЕДКОМ)
    # ============================================================
    
    def _pattern_to_regex(self, pattern: str) -> re.Pattern:
        """Конвертирует паттерн из ТендерПлана в regex.
        емкост* -> емкост.*
        пакет* мед* отход* -> пакет.* мед.* отход.*
        """
        # Экранируем спецсимволы кроме *
        escaped = re.escape(pattern).replace(r'\*', '.*')
        return re.compile(escaped, re.IGNORECASE)
    
    def _check_pattern_match(self, text: str, pattern: str) -> bool:
        """Проверяет совпадение текста с паттерном."""
        try:
            regex = self._pattern_to_regex(pattern)
            return bool(regex.search(text))
        except:
            return False
    
    def _calculate_relevance(self, tender: Tender, product: dict) -> int:
        """
        Считает релевантность тендера с учётом КОНТЕКСТА.
        
        Механика:
        1. Контекстные пары (обязательные комбинации) — высокий вес
        2. Одиночные ключевые слова — средний вес  
        3. Негативный контекст — штраф
        4. Медицинский контекст — бонус
        """
        text = f"{tender.title} {tender.description}".lower()
        
        score = 0
        max_score = 100
        matches_detail = []  # Для отладки
        
        # ═══════════════════════════════════════════════════════════
        # 1. КОНТЕКСТНЫЕ ПАРЫ (обязательные комбинации) — вес 25 баллов
        # ═══════════════════════════════════════════════════════════
        context_pairs = product.get('context_pairs', [])
        if not context_pairs:
            # Дефолтные пары для медицинской тематики
            context_pairs = [
                ("контейнер", "медицинск"),
                ("контейнер", "отход"),
                ("контейнер", "биологическ"),
                ("емкость", "медицинск"),
                ("емкость", "отход"),
                ("емкость", "биоматериал"),
                ("пакет", "отход"),
                ("пакет", "утилизац"),
                ("баночк", "анализ"),
                ("баночк", "биоматериал"),
                ("контейнер", "класс б"),
                ("контейнер", "класс а"),
            ]
        
        for word1, word2 in context_pairs:
            if word1 in text and word2 in text:
                score += 25
                matches_detail.append(f"пара:{word1}+{word2}")
                break  # Одна пара = достаточно для понимания контекста
        
        # ═══════════════════════════════════════════════════════════
        # 2. ОДИНОЧНЫЕ КЛЮЧЕВЫЕ СЛОВА — вес 5-10 баллов
        # ═══════════════════════════════════════════════════════════
        keywords = product.get('keywords', [])
        keyword_matches = 0
        
        for keyword in keywords:
            if self._check_pattern_match(text, keyword):
                keyword_matches += 1
                if keyword_matches <= 5:  # Первые 5 совпадений дают по 10
                    score += 10
                else:  # Остальные по 5
                    score += 5
        
        # ═══════════════════════════════════════════════════════════
        # 3. МЕДИЦИНСКИЙ КОНТЕКСТ — бонус +15
        # ═══════════════════════════════════════════════════════════
        medical_markers = [
            "медицинск", "медотход", "лечебн", "больниц", "поликлиник",
            "здравоохранен", "лпу", "гбуз", "гауз", "мбуз", "фгбу",
            "класс а", "класс б", "класс в", "класс г",
            "биоматериал", "биологическ", "анализ", "лаборатор",
            "санпин", "стерил", "дезинфекц"
        ]
        
        medical_found = sum(1 for m in medical_markers if m in text)
        if medical_found >= 2:
            score += 15
            matches_detail.append(f"мед.контекст:{medical_found}")
        
        # ═══════════════════════════════════════════════════════════
        # 4. НЕГАТИВНЫЙ КОНТЕКСТ — штраф -30/-50
        # ═══════════════════════════════════════════════════════════
        negative_contexts = product.get('negative_context', [])
        if not negative_contexts:
            # Дефолтные негативные контексты
            negative_contexts = [
                # Явно НЕ наше (полный штраф -50)
                ("контейнер", "мусор", -50),
                ("контейнер", "бытов", -50),
                ("контейнер", "тбо", -50),
                ("контейнер", "строительн", -50),
                ("емкость", "топлив", -50),
                ("емкость", "вод", -40),
                ("пакет", "пищев", -50),
                ("пакет", "фасовоч", -50),
                ("пакет", "мусор", -50),
                ("баночк", "пищев", -50),
                # Сомнительно (частичный штраф -30)
                ("контейнер", "хранен", -30),
                ("пакет", "упаков", -30),
            ]
        
        for item in negative_contexts:
            if len(item) == 3:
                word1, word2, penalty = item
                if word1 in text and word2 in text:
                    score += penalty  # penalty уже отрицательный
                    matches_detail.append(f"негатив:{word1}+{word2}={penalty}")
        
        # ═══════════════════════════════════════════════════════════
        # 5. ФИНАЛЬНЫЙ РАСЧЁТ
        # ═══════════════════════════════════════════════════════════
        # Нормализуем в диапазон 0-100
        relevance = max(0, min(score, max_score))
        
        # Логируем детали для отладки
        if matches_detail:
            logger.debug(f"Релевантность {relevance}%: {tender.title[:50]} | {matches_detail}")
        
        return relevance
    
    def _check_exclusions(self, tender: Tender, exclusions: List[str] = None) -> bool:
        """Проверяет тендер на исключения. Возвращает True если надо исключить."""
        if not CONFIG_LOADED:
            return False
            
        all_exclusions = exclusions or []
        
        # Добавляем глобальные исключения из конфига
        if hasattr(config, 'EXCLUSIONS'):
            all_exclusions = list(set(all_exclusions + config.EXCLUSIONS))
        
        text = f"{tender.title} {tender.description}".lower()
        
        for excl in all_exclusions:
            if self._check_pattern_match(text, excl):
                logger.debug(f"Исключён по паттерну '{excl}': {tender.title[:50]}")
                return True
        
        return False
    
    def _check_red_flags(self, tender: Tender) -> List[str]:
        """Проверяет тендер на красные флаги. Возвращает список найденных."""
        if not CONFIG_LOADED or not hasattr(config, 'RED_FLAGS'):
            return []
            
        text = f"{tender.title} {tender.description}".lower()
        found = []
        
        for flag in config.RED_FLAGS:
            if flag.lower() in text:
                found.append(flag)
        
        return found
    
    def _get_all_keywords(self) -> List[str]:
        """Возвращает все ключевые слова из конфига для поиска."""
        if not CONFIG_LOADED or not hasattr(config, 'PRODUCTS'):
            return []
        
        keywords = []
        for product in config.PRODUCTS:
            # Берём первые 3 самых важных ключевых слова из каждого продукта
            product_keywords = product.get('keywords', [])[:3]
            keywords.extend(product_keywords)
        
        return keywords
    
    def search_by_config(self, region: str = "", limit: int = 50, 
                         product_id: str = None) -> List[Tender]:
        """Умный поиск по конфигурации МЕДКОМ.
        
        Args:
            region: Регион поиска (пусто = вся Россия)
            limit: Максимум тендеров
            product_id: ID продукта из конфига (None = все продукты)
        
        Returns:
            Список тендеров отсортированный по релевантности
        """
        if not CONFIG_LOADED:
            print("[!] config.py не загружен!")
            return []
        
        products = config.PRODUCTS
        if product_id:
            products = [p for p in products if p['id'] == product_id]
            if not products:
                print(f"[!] Продукт '{product_id}' не найден в конфиге")
                return []
        
        all_tenders = {}  # number -> Tender (для дедупликации)
        
        for product in products:
            print(f"\n🔍 Поиск: {product['name']}")
            
            # Берём ключевые слова для поиска (упрощённые, без *)
            search_keywords = []
            for kw in product.get('keywords', [])[:5]:
                # Убираем * и берём первое слово для поиска
                clean_kw = kw.replace('*', '').split()[0]
                if len(clean_kw) >= 3:
                    search_keywords.append(clean_kw)
            
            for keyword in search_keywords[:3]:  # Максимум 3 запроса на продукт
                print(f"  → Ключ: {keyword}")
                
                tenders = self.search(
                    region=region,
                    industry="медицина",  # Фиксированная отрасль
                    keyword=keyword,
                    limit=limit // len(products)
                )
                
                for t in tenders:
                    # Пропускаем если уже есть
                    if t.number in all_tenders:
                        continue
                    
                    # Проверяем исключения
                    product_excl = product.get('exclusions', [])
                    if self._check_exclusions(t, product_excl):
                        continue
                    
                    # Считаем релевантность
                    t.relevance = self._calculate_relevance(t, product)
                    t.matched_product = product['name']
                    
                    # Проверяем красные флаги
                    t.red_flags = self._check_red_flags(t)
                    
                    all_tenders[t.number] = t
        
        # Сортируем по релевантности
        result = sorted(all_tenders.values(), key=lambda x: x.relevance, reverse=True)
        
        # Фильтруем по минимальной релевантности
        min_rel = 0
        if hasattr(config, 'SEARCH_SETTINGS'):
            min_rel = config.SEARCH_SETTINGS.get('min_relevance', 0)
        
        if min_rel > 0:
            result = [t for t in result if t.relevance >= min_rel]
        
        print(f"\n✅ Найдено {len(result)} релевантных тендеров")
        return result[:limit]
    
    def get_products_list(self) -> List[dict]:
        """Возвращает список продуктов из конфига."""
        if not CONFIG_LOADED or not hasattr(config, 'PRODUCTS'):
            return []
        return [{'id': p['id'], 'name': p['name']} for p in config.PRODUCTS]

    def get_details(self, url: str) -> dict:
        details = {'documents': [], 'requirements': '', 'description': ''}
        try:
            resp = self.session.get(url)
            soup = BeautifulSoup(resp.text, 'html.parser')
            html = resp.text

            if desc_div := soup.find('div', class_='description'):
                details['description'] = desc_div.get_text(separator=' ', strip=True)[:3000]
            
            if req_span := soup.find('span', string=re.compile(r'Требования', re.I)):
                if parent := req_span.find_parent('div'):
                    details['requirements'] = parent.get_text(separator='\n', strip=True)[:5000]

            if m := re.search(r'var\s+tendersData\s*=\s*(\{[^;]+\})', html):
                try:
                    data = json.loads(m.group(1))
                    for k in data:
                        for f in data[k].get('files_by_date', {}).values():
                            for x in f:
                                if x.get('link'):
                                    details['documents'].append(TenderFile(
                                        id=str(x.get('id', '')), title=x.get('title', 'file'),
                                        url=x['link'], extension=x.get('extension', '')
                                    ))
                except: pass
            
            if not details['documents']:
                for item in soup.select('.tender-files__item'):
                    if a := item.select_one('a[href]'):
                        details['documents'].append(TenderFile(
                            id='', title=item.get_text(strip=True), url=a['href'], extension=''
                        ))

        except Exception as e:
            logger.debug(f"Details error: {e}")
            pass
        return details

    def download_files(self, files: List[TenderFile], folder: str) -> str:
        target = self.downloads_dir / folder
        target.mkdir(exist_ok=True)
        count = 0
        
        for f in files:
            if not f.url: continue
            
            safe_title = re.sub(r'[^\w\-\.\s]', '_', f.title).strip()
            print(f"Скачивание: {safe_title}")
            
            name = f"{f.id}_{safe_title}"
            path = target / name
            
            try:
                headers = {'Referer': self.BASE_URL} if 'rostender' in f.url else {}
                with self.session.get(f.url, headers=headers, stream=True, timeout=60) as r:

                    content_type = r.headers.get('Content-Type', '').lower()
                    if 'html' in content_type:
                        if not path.suffix:
                            path = path.with_suffix('.html')
                    
                    r.raise_for_status()
                    
                    with open(path, 'wb') as out:
                        for chunk in r.iter_content(8192): out.write(chunk)
                
                f.local_path = str(path)

                if path.suffix == '.zip':
                    try:
                        with zipfile.ZipFile(path, 'r') as z: 
                            extract_path = target / f"{f.id}_extracted"
                            z.extractall(extract_path)
                            print(f"  -> Распакован в {extract_path.name}")
                    except Exception:
                        print("  -> Ошибка распаковки ZIP")

                count += 1
            except Exception as e:
                logger.debug(f"Download error {f.title}: {e}")
                print(f"  -> Ошибка: {e}")
                pass
            
        print(f"Скачано файлов: {count}/{len(files)}")
        return str(target) if count > 0 else ""


# ============================================================
# УМНЫЙ ПОИСК С РЕЛЕВАНТНОСТЬЮ
# ============================================================
class SmartSearch:
    """
    Умный поиск тендеров с:
    - Поиском по всем ключевым словам продукции
    - Фильтрацией по исключениям
    - AI-оценкой релевантности
    - Проверкой красных флагов
    """
    
    def __init__(self, openai_key: str = None):
        self.parser = TenderParser()
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY")
        
        # Загружаем конфиг
        try:
            from config import PRODUCTS, EXCLUSIONS, RED_FLAGS, SEARCH_SETTINGS, COMPANY
            self.products = PRODUCTS
            self.exclusions = EXCLUSIONS
            self.red_flags = RED_FLAGS
            self.settings = SEARCH_SETTINGS
            self.company = COMPANY
        except ImportError:
            logger.warning("config.py not found, using defaults")
            self.products = []
            self.exclusions = []
            self.red_flags = []
            self.settings = {"min_relevance": 30, "ai_score_relevance": False}
            self.company = {}
    
    def get_all_keywords(self) -> List[str]:
        """Собирает все ключевые слова из всех продуктов"""
        keywords = []
        for product in self.products:
            keywords.extend(product.get("keywords", []))
        return list(set(keywords))  # Убираем дубликаты
    
    def is_excluded(self, tender: Tender) -> bool:
        """Проверяет, должен ли тендер быть исключён"""
        text = f"{tender.title} {tender.description}".lower()
        for excl in self.exclusions:
            if excl.lower() in text:
                return True
        return False
    
    def check_red_flags(self, tender: Tender) -> List[str]:
        """Находит красные флаги в тендере"""
        text = f"{tender.title} {tender.description}".lower()
        found = []
        for flag in self.red_flags:
            if flag.lower() in text:
                found.append(flag)
        return found
    
    def calculate_keyword_score(self, tender: Tender) -> tuple:
        """
        Улучшенная оценка по ключевым словам с учётом КОНТЕКСТА
        
        Возвращает: (score: int 0-100, comment: str)
        """
        text = f"{tender.title} {tender.description}".lower()
        
        total_score = 0
        matched_products = []
        reasons = []
        global_negative = False  # Флаг глобального негатива
        negative_reasons = []
        
        for product in self.products:
            product_score = 0
            product_name = product.get("name", "")
            
            # 1. Базовые ключевые слова (упрощённые для поиска)
            keywords = product.get("keywords", [])
            keyword_matches = 0
            for kw in keywords:
                # Убираем * и ~ из синтаксиса ТендерПлана
                simple_kw = kw.replace("*", "").split()[0] if kw else ""
                if simple_kw and simple_kw.lower() in text:
                    keyword_matches += 1
            
            if keyword_matches > 0:
                product_score += min(30, keyword_matches * 5)  # До 30 баллов за слова
            
            # 2. КОНТЕКСТНЫЕ ПАРЫ (+25 за каждую пару)
            context_pairs = product.get("context_pairs", [])
            context_matches = 0
            for word1, word2 in context_pairs:
                if word1.lower() in text and word2.lower() in text:
                    context_matches += 1
            
            if context_matches > 0:
                bonus = min(50, context_matches * 25)  # До 50 баллов
                product_score += bonus
                reasons.append(f"{product_name}: +{context_matches} пар")
            
            # 3. НЕГАТИВНЫЙ КОНТЕКСТ (обнуляет совпадение!)
            negative_context = product.get("negative_context", [])
            has_negative = False
            for item in negative_context:
                if len(item) >= 3:
                    word1, word2, penalty = item[0], item[1], item[2]
                    if word1.lower() in text and word2.lower() in text:
                        has_negative = True
                        global_negative = True
                        negative_reasons.append(f"{word1}+{word2}")
            
            # Если негативный контекст - этот продукт НЕ подходит
            if has_negative:
                product_score = 0
            
            if product_score > 0:
                total_score += product_score
                matched_products.append(product_name)
        
        # Если был глобальный негатив - сильно снижаем общий балл
        if global_negative:
            total_score = max(0, total_score - 50)
        
        # Нормализуем до 100
        final_score = max(0, min(100, total_score))
        
        # Формируем комментарий
        comment = ""
        if matched_products:
            comment = f"Подходит: {', '.join(matched_products[:2])}"
        if reasons:
            comment += f" ({'; '.join(reasons[:2])})"
        if negative_reasons:
            comment = f"⚠️ Сомнительно: {', '.join(negative_reasons[:2])}"
        
        return final_score, comment.strip()
    
    async def ai_score_relevance(self, tenders: List[Tender]) -> List[dict]:
        """AI оценивает релевантность тендеров для компании с пониманием КОНТЕКСТА"""
        if not self.openai_key or not tenders:
            results = []
            for t in tenders:
                score, comment = self.calculate_keyword_score(t)
                results.append({"tender": t, "relevance": score, "ai_comment": comment})
            return results
        
        import httpx
        
        # Описание компании и продукции
        company_name = self.company.get("name", "Компания")
        company_desc = self.company.get("description", "")
        
        # Детальное описание продуктов с примерами
        products_desc = ""
        for p in self.products:
            products_desc += f"\n• {p['name']}:\n"
            products_desc += f"  Ключевые слова: {', '.join(p.get('keywords', [])[:10])}\n"
            if p.get('exclusions'):
                products_desc += f"  НЕ подходит если: {', '.join(p.get('exclusions', [])[:5])}\n"
        
        # Формируем список тендеров для оценки
        tenders_text = "\n".join([
            f"{i+1}. [{t.number}] {t.title}"
            for i, t in enumerate(tenders[:self.settings.get("max_ai_scoring", 20)])
        ])
        
        prompt = f"""Ты эксперт по госзакупкам. Оцени релевантность тендеров для компании-производителя.

КОМПАНИЯ: {company_name}
СПЕЦИАЛИЗАЦИЯ: {company_desc}

ПРОДУКЦИЯ КОМПАНИИ:{products_desc}

ВАЖНО - ПРАВИЛА ОЦЕНКИ КОНТЕКСТА:
1. "Контейнер для строительного мусора" — НЕ подходит (не медицинский)
2. "Пакет документов" — НЕ подходит (не физический пакет)
3. "Медицинское оборудование МРТ/УЗИ" — НЕ подходит (это техника, не расходники)
4. "Емкость для сбора биоматериала" — ПОДХОДИТ
5. "Контейнер для острого инструментария" — ПОДХОДИТ
6. "Пакет для утилизации отходов класса Б" — ПОДХОДИТ

ТЕНДЕРЫ ДЛЯ ОЦЕНКИ:
{tenders_text}

Для каждого тендера оцени:
- Релевантность (0-100%):
  * 80-100 = точное совпадение с продукцией компании
  * 60-79 = высокая вероятность (похожая продукция)
  * 40-59 = средняя (возможно подходит)
  * 20-39 = низкая (скорее не подходит)
  * 0-19 = не подходит совсем

Формат ответа (JSON массив):
[
  {{"num": 1, "relevance": 85, "comment": "Контейнеры для медотходов - прямое совпадение"}},
  {{"num": 2, "relevance": 15, "comment": "Строительные контейнеры - не наш профиль"}}
]

Только JSON, без пояснений."""

        try:
            # Используем proxyapi если настроен
            api_url = os.getenv("OPENAI_BASE_URL", "https://api.proxyapi.ru/openai/v1")
            if not api_url.endswith("/chat/completions"):
                api_url = api_url.rstrip("/") + "/chat/completions"
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    api_url,
                    headers={"Authorization": f"Bearer {self.openai_key}"},
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1
                    }
                )
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                
                # Парсим JSON
                import json
                # Убираем markdown если есть
                if "```" in content:
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                
                scores = json.loads(content.strip())
                
                # Применяем оценки
                results = []
                for i, tender in enumerate(tenders):
                    score_data = next((s for s in scores if s.get("num") == i + 1), None)
                    if score_data:
                        results.append({
                            "tender": tender,
                            "relevance": score_data.get("relevance", 50),
                            "ai_comment": score_data.get("comment", "")
                        })
                    else:
                        score, comment = self.calculate_keyword_score(tender)
                        results.append({
                            "tender": tender,
                            "relevance": score,
                            "ai_comment": comment
                        })
                
                return results
                
        except Exception as e:
            logger.error(f"AI scoring error: {e}")
            results = []
            for t in tenders:
                score, comment = self.calculate_keyword_score(t)
                results.append({"tender": t, "relevance": score, "ai_comment": comment})
            return results
    
    async def search(
        self,
        product_id: str = None,  # ID продукта из конфига или None для всех
        region: str = "",
        limit: int = 50,
        use_ai: bool = True
    ) -> List[dict]:
        """
        Умный поиск тендеров
        
        Возвращает список словарей:
        {
            "tender": Tender,
            "relevance": int (0-100),
            "ai_comment": str,
            "red_flags": List[str],
            "product_match": str  # Какой продукт подходит
        }
        """
        # Определяем ключевые слова для поиска
        if product_id:
            product = next((p for p in self.products if p.get("id") == product_id), None)
            if product:
                keywords = product.get("keywords", [])[:5]  # Берём топ-5 для поиска
            else:
                keywords = self.get_all_keywords()[:5]
        else:
            keywords = self.get_all_keywords()[:10]
        
        # Собираем результаты по всем ключевым словам
        all_tenders = {}
        for kw in keywords:
            try:
                results = await self.parser.search(
                    region=region,
                    keyword=kw,
                    limit=limit // len(keywords) + 5
                )
                for t in results:
                    if t.number not in all_tenders:
                        all_tenders[t.number] = t
            except Exception as e:
                logger.debug(f"Search error for '{kw}': {e}")
                continue
        
        tenders = list(all_tenders.values())
        logger.info(f"Found {len(tenders)} unique tenders")
        
        # Фильтруем исключения
        tenders = [t for t in tenders if not self.is_excluded(t)]
        logger.info(f"After exclusions: {len(tenders)} tenders")
        
        # Оцениваем релевантность
        if use_ai and self.settings.get("ai_score_relevance"):
            scored = await self.ai_score_relevance(tenders)
        else:
            scored = []
            for t in tenders:
                score, comment = self.calculate_keyword_score(t)
                scored.append({"tender": t, "relevance": score, "ai_comment": comment})
        
        # Добавляем красные флаги и определяем продукт
        results = []
        for item in scored:
            tender = item["tender"]
            red_flags = self.check_red_flags(tender)
            
            # Определяем какой продукт подходит
            product_match = ""
            text = f"{tender.title} {tender.description}".lower()
            for product in self.products:
                for kw in product.get("keywords", []):
                    if kw.lower() in text:
                        product_match = product.get("name", "")
                        break
                if product_match:
                    break
            
            results.append({
                "tender": tender,
                "relevance": item["relevance"],
                "ai_comment": item.get("ai_comment", ""),
                "red_flags": red_flags,
                "product_match": product_match
            })
        
        # Сортируем по релевантности
        results.sort(key=lambda x: x["relevance"], reverse=True)
        
        # Фильтруем по минимальной релевантности
        min_rel = self.settings.get("min_relevance", 30)
        if not self.settings.get("show_unscored", True):
            results = [r for r in results if r["relevance"] >= min_rel]
        
        return results[:limit]


# Функция для использования из API
async def smart_search(
    product_id: str = None,
    region: str = "",
    limit: int = 50,
    use_ai: bool = True,
    openai_key: str = None
) -> List[dict]:
    """Обёртка для умного поиска"""
    searcher = SmartSearch(openai_key)
    return await searcher.search(product_id, region, limit, use_ai)
