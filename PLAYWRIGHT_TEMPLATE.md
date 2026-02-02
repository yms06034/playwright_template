# Playwright 크롤러 템플릿

봇 탐지 우회, 사람처럼 행동, 병렬 크롤링을 지원하는 Playwright 기반 크롤러 프레임워크입니다.

## 목차

- [설치](#설치)
- [빠른 시작](#빠른-시작)
- [설정 (CrawlerConfig)](#설정-crawlerconfig)
- [크롤러 클래스](#크롤러-클래스)
- [병렬 크롤링](#병렬-크롤링)
- [리소스 최적화](#리소스-최적화)
- [유틸리티](#유틸리티)
- [실전 예제](#실전-예제)
- [트러블슈팅](#트러블슈팅)

---

## 설치

```bash
# 필수 패키지
pip install playwright

# Playwright 브라우저 설치
playwright install

# (선택) 시스템 리소스 조회용
pip install psutil
```

**requirements.txt:**
```
playwright>=1.40.0
```

---

## 빠른 시작

### 1. 기본 크롤러 만들기

```python
import asyncio
from playwright_template import BaseCrawler, CrawlerConfig

class MyCrawler(BaseCrawler):
    async def run(self, url: str):
        # 페이지 이동
        await self.goto(url)

        # 데이터 추출
        title = await self.page.title()

        return {"title": title}

async def main():
    config = CrawlerConfig(mode="fast", headless=True)
    crawler = MyCrawler(config)
    result = await crawler.execute("https://example.com")
    print(result)

asyncio.run(main())
```

### 2. 설정 없이 바로 사용

```python
crawler = MyCrawler()  # 기본 설정 (balanced 모드)
result = await crawler.execute("https://example.com")
```

---

## 설정 (CrawlerConfig)

### 기본 모드

| 모드 | 설명 | 속도 | 안전성 |
|------|------|------|--------|
| `fast` | 빠른 크롤링 | 빠름 | 낮음 |
| `balanced` | 균형 (기본값) | 보통 | 보통 |
| `stealth` | 스텔스 모드 | 느림 | 높음 |

```python
config = CrawlerConfig(mode="stealth")
```

### 전체 설정 옵션

```python
from playwright_template import CrawlerConfig

config = CrawlerConfig(
    # === 기본 설정 ===
    mode="balanced",              # "fast", "balanced", "stealth"

    # === 브라우저 설정 ===
    browser="chromium",           # "chromium", "firefox", "webkit"
    headless=True,                # True: 백그라운드, False: 브라우저 보임
    viewport_width=1920,          # 뷰포트 너비
    viewport_height=1080,         # 뷰포트 높이

    # === 지역/언어 설정 ===
    locale="ko-KR",               # 로케일
    timezone="Asia/Seoul",        # 시간대
    languages=["ko-KR", "ko", "en-US", "en"],  # navigator.languages
    accept_language="ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",  # HTTP 헤더

    # === 딜레이 설정 (ms) ===
    min_delay=300,                # 최소 딜레이
    max_delay=800,                # 최대 딜레이
    click_delay=500,              # 클릭 후 딜레이
    scroll_delay=300,             # 스크롤 후 딜레이

    # === 행동 시뮬레이션 ===
    simulate_reading=True,        # 페이지 읽는 척
    use_human_mouse=False,        # 자연스러운 마우스 움직임

    # === 타임아웃 (ms) ===
    navigation_timeout=30000,     # 페이지 로드 타임아웃
    action_timeout=5000,          # 동작 타임아웃

    # === 재시도 설정 ===
    max_retries=3,                # 최대 재시도 횟수
    retry_delay=1000,             # 재시도 간 딜레이 (ms)

    # === 리소스 차단 ===
    block_images=False,           # 이미지 차단
    block_fonts=False,            # 폰트 차단
    block_ads=False,              # 광고/추적 차단
    block_media=False,            # 비디오/오디오 차단
    block_stylesheets=False,      # CSS 차단

    # === 프록시 ===
    proxy=None,                   # "http://ip:port" 또는 "http://user:pass@ip:port"

    # === 쿠키/세션 ===
    cookies_file=None,            # 쿠키 저장/로드 경로

    # === Rate Limiting ===
    requests_per_minute=0,        # 분당 요청 제한 (0=무제한)

    # === 디버그 ===
    debug=False,                  # 디버그 모드
    debug_dir="debug",            # 디버그 파일 저장 폴더
    verbose=False,                # 상세 로그
    debug_video_width=1280,       # 녹화 영상 너비
    debug_video_height=720,       # 녹화 영상 높이
    log_file=None,                # 로그 파일 경로

    # === 사용자 정의 확장 ===
    custom_ad_domains=[],         # 추가 광고 도메인
    custom_captcha_indicators=[], # 추가 캡차 감지 키워드
    custom_popup_selectors=[],    # 추가 팝업 닫기 셀렉터
    custom_browser_args=[],       # 추가 브라우저 실행 옵션

    # === 마우스 움직임 (stealth 모드) ===
    mouse_move_steps_min=10,      # 마우스 이동 단계 최소
    mouse_move_steps_max=25,      # 마우스 이동 단계 최대
    mouse_noise_range=5,          # 마우스 움직임 노이즈 범위 (px)

    # === 타이핑 ===
    typing_delay_min=50,          # 타이핑 딜레이 최소 (ms)
    typing_delay_max=150,         # 타이핑 딜레이 최대 (ms)
    typing_pause_probability=0.1, # 타이핑 중 멈춤 확률

    # === 스크롤 ===
    scroll_steps_min=3,           # 스크롤 단계 최소
    scroll_steps_max=6,           # 스크롤 단계 최대

    # === HTTP ===
    user_agent=None,              # User-Agent (None=자동)
    extra_headers={},             # 추가 HTTP 헤더
)
```

### 설정 파일에서 로드

**config.json:**
```json
{
    "mode": "fast",
    "browser": "chromium",
    "headless": true,
    "locale": "en-US",
    "timezone": "America/New_York",
    "languages": ["en-US", "en"]
}
```

```python
from playwright_template import load_config_from_file

config = load_config_from_file("config.json")
```

### 환경변수에서 로드

```bash
export CRAWLER_MODE=fast
export CRAWLER_BROWSER=chromium
export CRAWLER_HEADLESS=true
export CRAWLER_PROXY=http://proxy:8080
```

```python
from playwright_template import load_config_from_env

config = load_config_from_env()
```

---

## 크롤러 클래스

### BaseCrawler

모든 크롤러의 기본 클래스입니다.

```python
from playwright_template import BaseCrawler, CrawlerConfig

class MyCrawler(BaseCrawler):
    async def run(self, url: str):
        # 페이지 이동
        await self.goto(url)

        # 스크린샷 (debug=True일 때만)
        await self.screenshot("step1")

        # 텍스트로 클릭
        await self.click_by_text("로그인")

        # 셀렉터로 클릭
        await self.wait_and_click("button.submit")

        # JavaScript 실행
        data = await self.page.evaluate('''
            () => {
                return {
                    title: document.title,
                    items: [...document.querySelectorAll('.item')].map(el => el.textContent)
                };
            }
        ''')

        return data
```

#### 주요 메서드

| 메서드 | 설명 |
|--------|------|
| `goto(url)` | 페이지 이동 |
| `screenshot(name)` | 스크린샷 (디버그 모드) |
| `click_by_text(text)` | 텍스트로 요소 클릭 |
| `click_by_role(role, name)` | role 속성으로 클릭 |
| `wait_and_click(selector)` | 대기 후 클릭 |
| `extract_text(selector)` | 텍스트 추출 |
| `extract_attribute(selector, attr)` | 속성 추출 |
| `infinite_scroll(max_scrolls)` | 무한 스크롤 |
| `close_popups()` | 팝업 닫기 |
| `fill_form(fields)` | 폼 입력 |
| `detect_captcha()` | 캡차 감지 |
| `get_all_links()` | 모든 링크 추출 |
| `save_html(filename)` | HTML 저장 |
| `save_json(data, filename)` | JSON 저장 |

### LightweightCrawler

리소스 최적화된 경량 크롤러입니다. 이미지, 폰트, 미디어를 기본 차단합니다.

```python
from playwright_template import LightweightCrawler

class FastCrawler(LightweightCrawler):
    async def run(self, url: str):
        await self.goto(url)  # domcontentloaded로 빠른 로드
        return await self.page.title()
```

**주의:** 이미지 차단 시 프로필 이미지의 `alt` 속성을 가져올 수 없을 수 있습니다. 필요하면 `BaseCrawler`를 사용하세요.

```python
# 이미지 필요한 경우
config = CrawlerConfig(block_images=False)
crawler = MyLightweightCrawler(config)
```

### ContextReuseCrawler

같은 도메인의 여러 페이지를 크롤링할 때 효율적입니다.

```python
from playwright_template import ContextReuseCrawler, CrawlerConfig

class MultiPageCrawler(ContextReuseCrawler):
    async def run(self, url: str):
        await self.goto(url)
        return await self.page.title()

async def main():
    config = CrawlerConfig(mode="fast")

    async with MultiPageCrawler(config) as crawler:
        for url in urls:
            result = await crawler.crawl(url)
            print(result)

    # 또는
    crawler = MultiPageCrawler(config)
    await crawler.start()

    for url in urls:
        result = await crawler.crawl(url)

    await crawler.stop()
```

### DebugCrawler

대화형 디버그 모드로 셀렉터를 테스트합니다.

```python
from playwright_template import DebugCrawler

async def debug():
    crawler = DebugCrawler()
    await crawler.start("https://example.com")

    # 셀렉터 테스트
    result = await crawler.test_selector("h1")
    print(result)
    # {'selector': 'h1', 'count': 1, 'texts': ['Example'], 'visible': 1}

    # 요소 하이라이트
    await crawler.highlight("h1", color="red")

    # 페이지 정보
    info = await crawler.get_page_info()
    print(info)

    # 직접 조작
    await crawler.page.click("button")

    await crawler.stop()
```

---

## 병렬 크롤링

### 간단한 병렬 크롤링

```python
from playwright_template import crawl_parallel, CrawlerConfig

class MyCrawler(BaseCrawler):
    async def run(self, url: str):
        await self.goto(url)
        return {"title": await self.page.title()}

async def main():
    urls = [
        "https://example1.com",
        "https://example2.com",
        "https://example3.com",
    ]

    config = CrawlerConfig(mode="fast")
    results = await crawl_parallel(
        crawler_class=MyCrawler,
        urls=urls,
        config=config,
        max_concurrent=3  # 동시 실행 수
    )

    for r in results:
        if r["success"]:
            print(f"성공: {r['url']} - {r['data']}")
        else:
            print(f"실패: {r['url']} - {r['error']}")
```

### CrawlerPool (고급)

Queue 기반 워커 풀로 대규모 크롤링을 처리합니다.

```python
from playwright_template import CrawlerPool, CrawlerConfig, CrawlTask

async def main():
    config = CrawlerConfig(mode="fast", headless=True)

    pool = CrawlerPool(
        crawler_class=MyCrawler,
        config=config,
        num_workers="auto",           # 시스템에 맞게 자동 조절
        reuse_browsers=True,          # 브라우저 재사용
        proxies=[                     # Round-Robin 프록시
            "http://proxy1:8080",
            "http://proxy2:8080",
        ],
        browsers=["chromium", "firefox"],  # Round-Robin 브라우저
    )

    # 작업 추가
    pool.add_urls([
        "https://example1.com",
        "https://example2.com",
    ])

    # 우선순위 지정
    pool.add_url("https://important.com", priority=10)  # 먼저 처리

    # 콜백과 함께 실행
    async def on_complete(result):
        status = "O" if result.success else "X"
        print(f"{status} {result.url} ({result.elapsed_time:.1f}초)")

    def on_progress(completed, failed, total):
        print(f"진행: {completed + failed}/{total}")

    results = await pool.run(
        on_complete=on_complete,
        on_progress=on_progress,
        progress_interval=10.0  # 10초마다 진행 상황 출력
    )

    # 통계
    stats = pool.get_stats()
    print(f"완료: {stats['completed']}, 실패: {stats['failed']}")
```

---

## 리소스 최적화

### ResourceOptimizer

```python
from playwright_template import ResourceOptimizer

# 페이지에 리소스 차단 적용
await ResourceOptimizer.setup_resource_blocking(
    page,
    block_images=True,
    block_fonts=True,
    block_media=True,
    block_ads=True,
    block_stylesheets=False,
    allowed_domains=["cdn.example.com"],  # 차단 제외
    custom_ad_domains=["my-ad.com"],      # 추가 광고 도메인
)

# 페이지 메모리 최적화
await ResourceOptimizer.optimize_page_memory(page)

# 최적의 워커 수 계산
workers = ResourceOptimizer.get_optimal_workers(
    total_urls=100,
    available_memory_gb=8.0,
    per_browser_memory_mb=150.0,
)
print(f"권장 워커: {workers}개")
```

### 기본값 변경

```python
# 클래스 변수로 기본값 조정
ResourceOptimizer.DEFAULT_BROWSER_MEMORY_MB = 200
ResourceOptimizer.DEFAULT_MAX_WORKERS = 15
ResourceOptimizer.DEFAULT_CPU_MULTIPLIER = 3

# 광고 도메인 추가
ResourceOptimizer.AD_TRACKING_DOMAINS.append("my-tracker.com")
```

### 시스템 리소스 확인

```python
from playwright_template import get_system_resources

info = get_system_resources()
print(f"CPU: {info['cpu_count']}코어")
print(f"메모리: {info['memory_available_gb']:.1f}GB")
print(f"권장 워커: {info['recommended_workers']}개")
```

---

## 유틸리티

### DataUtils

```python
from playwright_template import DataUtils

# 숫자 파싱 (천 단위 콤마, K/M 지원)
DataUtils.parse_number("1,234")     # 1234
DataUtils.parse_number("12.5K")     # 12500
DataUtils.parse_number("1.2M")      # 1200000

# 날짜 파싱
DataUtils.parse_date("2024-01-15 10:30:00")
DataUtils.parse_date("2024.01.15", formats=["%Y.%m.%d"])

# 텍스트 정리
DataUtils.clean_text("  여러   공백   정리  ")  # "여러 공백 정리"

# 추출
DataUtils.extract_urls(text)    # URL 목록
DataUtils.extract_emails(text)  # 이메일 목록
DataUtils.extract_phones(text)  # 전화번호 목록 (한국)
```

### Selectors

```python
from playwright_template import Selectors

# 셀렉터 생성 헬퍼
Selectors.by_text("로그인")                    # '*:has-text("로그인")'
Selectors.by_class_contains("comment")         # '*[class*="comment"]'
Selectors.by_id_contains("user")               # '*[id*="user"]'
Selectors.by_attr("data-id", "123")            # '*[data-id="123"]'
Selectors.by_attr_contains("href", "/user")    # '*[href*="/user"]'
Selectors.nth("li", 3)                         # 'li:nth-child(3)'
Selectors.first("li")                          # 'li:first-child'
Selectors.last("li")                           # 'li:last-child'
```

### 데코레이터

```python
from playwright_template import retry_on_failure, log_execution

@retry_on_failure(max_retries=3, delay=1.0)
async def unreliable_function():
    # 실패 시 3번까지 재시도
    pass

@log_execution
async def tracked_function():
    # 실행 시간 자동 로깅
    pass
```

### CSV 저장

```python
from playwright_template import save_to_csv

data = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25},
]

save_to_csv(data, "output.csv")  # Excel 호환 (utf-8-sig)
```

---

## 실전 예제

### 예제 1: 댓글 수집기

```python
import asyncio
from dataclasses import dataclass
from playwright_template import BaseCrawler, CrawlerConfig

@dataclass
class Comment:
    author: str
    content: str
    likes: int

class CommentCrawler(BaseCrawler):
    async def run(self, url: str):
        await self.goto(url)

        # 더보기 클릭
        await self.infinite_scroll(
            max_scrolls=10,
            load_more_selector="button:has-text('더보기')"
        )

        # 댓글 추출
        comments = await self.page.evaluate('''
            () => {
                return [...document.querySelectorAll('.comment')].map(el => ({
                    author: el.querySelector('.author')?.textContent || '',
                    content: el.querySelector('.content')?.textContent || '',
                    likes: parseInt(el.querySelector('.likes')?.textContent || '0')
                }));
            }
        ''')

        return [Comment(**c) for c in comments]

async def main():
    config = CrawlerConfig(
        mode="balanced",
        headless=True,
        block_fonts=True,
        block_ads=True,
    )

    crawler = CommentCrawler(config)
    comments = await crawler.execute("https://example.com/post/123")

    for c in comments:
        print(f"{c.author}: {c.content} ({c.likes})")

asyncio.run(main())
```

### 예제 2: 로그인 후 크롤링

```python
class AuthCrawler(BaseCrawler):
    async def login(self, username: str, password: str):
        await self.goto("https://example.com/login")

        # 폼 입력
        await self.fill_form({
            "#username": username,
            "#password": password,
        })

        # 로그인 버튼 클릭
        await self.click_by_text("로그인")

        # 로그인 완료 대기
        await self.wait_for_url("**/dashboard**")

    async def run(self, url: str):
        # 로그인
        await self.login("user", "pass")

        # 데이터 수집
        await self.goto(url)
        return await self.page.title()

# 쿠키 저장/재사용
config = CrawlerConfig(cookies_file="session.json")
```

### 예제 3: API 응답 가로채기

```python
class ApiCrawler(BaseCrawler):
    def __init__(self, config=None):
        super().__init__(config)
        self.api_data = []

    async def run(self, url: str):
        # API 응답 가로채기 설정
        async def capture_api(response):
            if "/api/data" in response.url:
                data = await response.json()
                self.api_data.append(data)

        await self.intercept_response("**/api/**", capture_api)

        # 페이지 이동 (API 호출 트리거)
        await self.goto(url)

        # 스크롤로 추가 데이터 로드
        await self.infinite_scroll(max_scrolls=5)

        return self.api_data
```

### 예제 4: 여러 브라우저로 테스트

```python
async def test_all_browsers():
    urls = ["https://example.com"]

    for browser in ["chromium", "firefox", "webkit"]:
        config = CrawlerConfig(browser=browser)
        crawler = MyCrawler(config)

        result = await crawler.execute(urls[0])
        print(f"{browser}: {result}")
```

### 예제 5: 프록시 로테이션

```python
from playwright_template import CrawlerPool

pool = CrawlerPool(
    crawler_class=MyCrawler,
    num_workers=5,
    proxies=[
        "http://proxy1.example.com:8080",
        "http://proxy2.example.com:8080",
        "http://proxy3.example.com:8080",
    ],
)

# 각 워커가 순환하며 프록시 사용
pool.add_urls(urls)
results = await pool.run()
```

---

## 트러블슈팅

### 캡차가 감지됨

```python
# stealth 모드 사용
config = CrawlerConfig(mode="stealth")

# 커스텀 캡차 키워드 추가
config = CrawlerConfig(
    custom_captcha_indicators=["verify-human", "보안확인"]
)
```

### 페이지 로드 타임아웃

```python
# 타임아웃 증가
config = CrawlerConfig(navigation_timeout=60000)

# networkidle 대신 load 사용
await self.page.goto(url, wait_until="load")

# 또는 특정 요소 대기
await self.page.goto(url, wait_until="domcontentloaded")
await self.page.wait_for_selector(".content", timeout=10000)
```

### 요소를 찾을 수 없음

```python
# 디버그 모드로 확인
config = CrawlerConfig(debug=True, headless=False)

# DebugCrawler로 셀렉터 테스트
crawler = DebugCrawler()
await crawler.start(url)
await crawler.test_selector("your-selector")
await crawler.highlight("your-selector")
```

### 메모리 부족

```python
# 리소스 차단
config = CrawlerConfig(
    block_images=True,
    block_fonts=True,
    block_media=True,
)

# 워커 수 제한
pool = CrawlerPool(crawler_class=MyCrawler, num_workers=3)

# 브라우저당 메모리 설정 조정
ResourceOptimizer.DEFAULT_BROWSER_MEMORY_MB = 200
```

### Rate Limiting

```python
# 분당 요청 제한
config = CrawlerConfig(requests_per_minute=30)

# 요청 간 딜레이 증가
config = CrawlerConfig(
    mode="stealth",
    min_delay=2000,
    max_delay=5000,
)
```

### 프로필 이미지 alt 속성이 안 잡힘

```python
# LightweightCrawler 대신 BaseCrawler 사용
# 또는 이미지 차단 비활성화
config = CrawlerConfig(block_images=False)
```

---

## 언어/지역 설정

### 영어 (미국)

```python
config = CrawlerConfig(
    locale="en-US",
    timezone="America/New_York",
    languages=["en-US", "en"],
    accept_language="en-US,en;q=0.9",
)
```

### 일본어

```python
config = CrawlerConfig(
    locale="ja-JP",
    timezone="Asia/Tokyo",
    languages=["ja-JP", "ja", "en-US", "en"],
    accept_language="ja-JP,ja;q=0.9,en-US;q=0.8,en;q=0.7",
)
```

### 중국어 (간체)

```python
config = CrawlerConfig(
    locale="zh-CN",
    timezone="Asia/Shanghai",
    languages=["zh-CN", "zh", "en-US", "en"],
    accept_language="zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7",
)
```

---

## 라이선스

MIT License
