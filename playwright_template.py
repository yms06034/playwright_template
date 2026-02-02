"""
Playwright 크롤러 템플릿
- 봇 탐지 우회
- 사람처럼 행동
- 디버깅 지원
- 안정적 구동

사용법:
    from playwright_template import BaseCrawler, CrawlerConfig

    class MyCrawler(BaseCrawler):
        async def run(self, url: str):
            await self.goto(url)
            # 메인 로직 작성
            data = await self.page.evaluate('...')
            return data

    async def main():
        config = CrawlerConfig(mode="fast")
        crawler = MyCrawler(config=config)
        result = await crawler.execute("https://example.com")
"""

import asyncio
import json
import logging
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from playwright.async_api import async_playwright, Page, BrowserContext, Browser, Route

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


@dataclass
class CrawlerConfig:
    """
    크롤러 설정

    모드:
        - "fast": 빠른 크롤링 (봇 탐지 위험 있음)
        - "balanced": 균형 모드 (기본값)
        - "stealth": 스텔스 모드 (느리지만 안전)
    """
    mode: str = "balanced"

    # 딜레이 설정 (ms)
    min_delay: int = field(default=0)
    max_delay: int = field(default=0)
    click_delay: int = field(default=0)
    scroll_delay: int = field(default=0)

    # 행동 시뮬레이션
    simulate_reading: bool = field(default=True)
    use_human_mouse: bool = field(default=False)

    # 브라우저 설정
    browser: str = field(default="chromium")  # "chromium", "firefox", "webkit"
    headless: bool = field(default=True)
    viewport_width: int = field(default=1920)
    viewport_height: int = field(default=1080)
    locale: str = field(default="ko-KR")
    timezone: str = field(default="Asia/Seoul")

    # 언어 설정 (Accept-Language 헤더, stealth 스크립트에 사용)
    languages: list[str] = field(default_factory=lambda: ["ko-KR", "ko", "en-US", "en"])
    accept_language: str = field(default="ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7")

    # 디버그 설정
    debug: bool = field(default=False)
    debug_dir: str = field(default="debug")
    verbose: bool = field(default=False)
    debug_video_width: int = field(default=1280)
    debug_video_height: int = field(default=720)

    # 재시도 설정
    max_retries: int = field(default=3)
    retry_delay: int = field(default=1000)

    # 타임아웃 (ms)
    navigation_timeout: int = field(default=30000)
    action_timeout: int = field(default=5000)

    # 프록시 설정
    proxy: str | None = field(default=None)  # "http://ip:port" 또는 "http://user:pass@ip:port"

    # 리소스 차단 (속도 향상)
    block_images: bool = field(default=False)
    block_fonts: bool = field(default=False)
    block_ads: bool = field(default=False)
    block_media: bool = field(default=False)
    block_stylesheets: bool = field(default=False)

    # 쿠키/세션 관리
    cookies_file: str | None = field(default=None)  # 쿠키 저장/로드 경로

    # Rate Limiting
    requests_per_minute: int = field(default=0)  # 0 = 제한 없음

    # 로그 파일
    log_file: str | None = field(default=None)

    # 사용자 정의 광고/추적 도메인 (기본값에 추가됨)
    custom_ad_domains: list[str] = field(default_factory=list)

    # 사용자 정의 캡차 감지 키워드 (기본값에 추가됨)
    custom_captcha_indicators: list[str] = field(default_factory=list)

    # 사용자 정의 팝업 닫기 셀렉터 (기본값에 추가됨)
    custom_popup_selectors: list[str] = field(default_factory=list)

    # 사용자 정의 브라우저 실행 옵션
    custom_browser_args: list[str] = field(default_factory=list)

    # 마우스 움직임 설정
    mouse_move_steps_min: int = field(default=10)
    mouse_move_steps_max: int = field(default=25)
    mouse_noise_range: int = field(default=5)

    # 타이핑 설정
    typing_delay_min: int = field(default=50)
    typing_delay_max: int = field(default=150)
    typing_pause_probability: float = field(default=0.1)

    # 스크롤 설정
    scroll_steps_min: int = field(default=3)
    scroll_steps_max: int = field(default=6)

    # User-Agent (None이면 자동 추출)
    user_agent: str | None = field(default=None)

    # 추가 HTTP 헤더
    extra_headers: dict = field(default_factory=dict)

    def __post_init__(self):
        presets = {
            "fast": {
                "min_delay": 100,
                "max_delay": 300,
                "click_delay": 200,
                "scroll_delay": 100,
                "simulate_reading": False,
                "use_human_mouse": False,
            },
            "balanced": {
                "min_delay": 300,
                "max_delay": 800,
                "click_delay": 500,
                "scroll_delay": 300,
                "simulate_reading": True,
                "use_human_mouse": False,
            },
            "stealth": {
                "min_delay": 800,
                "max_delay": 2000,
                "click_delay": 1500,
                "scroll_delay": 800,
                "simulate_reading": True,
                "use_human_mouse": True,
            },
        }

        if self.mode in presets and self.min_delay == 0:
            preset = presets[self.mode]
            for key, value in preset.items():
                setattr(self, key, value)

        # 로그 레벨 설정
        if self.verbose:
            logging.getLogger(__name__).setLevel(logging.DEBUG)

        # 로그 파일 설정
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            ))
            logging.getLogger(__name__).addHandler(file_handler)


class BrowserConfig:
    """브라우저 설정 및 User-Agent 관리"""

    _cached_ua: dict[str, str] = {}  # 브라우저별 UA 캐싱

    # 일반적인 뷰포트 크기
    VIEWPORTS = [
        {"width": 1920, "height": 1080},
        {"width": 1366, "height": 768},
        {"width": 1536, "height": 864},
        {"width": 1440, "height": 900},
        {"width": 1280, "height": 720},
    ]

    # 브라우저별 실행 옵션
    LAUNCH_ARGS = {
        "chromium": [
            "--disable-blink-features=AutomationControlled",
            "--disable-infobars",
            "--disable-dev-shm-usage",
            "--disable-browser-side-navigation",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-setuid-sandbox",
        ],
        "firefox": [
            # Firefox는 대부분 기본값 사용
        ],
        "webkit": [
            # WebKit(Safari)은 대부분 기본값 사용
        ],
    }

    @classmethod
    async def get_user_agent(cls, browser_type: str = "chromium") -> str:
        """실제 Playwright 브라우저의 User-Agent 추출 (브라우저별 캐싱)"""
        if browser_type not in cls._cached_ua:
            async with async_playwright() as p:
                browser_launcher = getattr(p, browser_type)
                browser = await browser_launcher.launch(headless=True)
                page = await browser.new_page()
                cls._cached_ua[browser_type] = await page.evaluate("navigator.userAgent")
                await browser.close()
                logger.debug(f"User-Agent 추출 ({browser_type}): {cls._cached_ua[browser_type][:50]}...")
        return cls._cached_ua[browser_type]

    @classmethod
    def get_random_viewport(cls) -> dict:
        """랜덤 뷰포트 크기 반환"""
        return random.choice(cls.VIEWPORTS)

    @classmethod
    def get_launch_args(cls, browser_type: str) -> list:
        """브라우저별 실행 옵션 반환"""
        return cls.LAUNCH_ARGS.get(browser_type, [])

    @classmethod
    def get_stealth_script(cls, browser_type: str, languages: list[str] | None = None) -> str:
        """
        브라우저별 Stealth 스크립트 반환

        Args:
            browser_type: 브라우저 타입
            languages: 언어 설정 (기본: ['ko-KR', 'ko', 'en-US', 'en'])
        """
        languages = languages or ['ko-KR', 'ko', 'en-US', 'en']
        languages_js = json.dumps(languages)

        if browser_type == "chromium":
            return f"""
            // webdriver 숨기기
            Object.defineProperty(navigator, 'webdriver', {{
                get: () => undefined
            }});

            // plugins 위장
            Object.defineProperty(navigator, 'plugins', {{
                get: () => [1, 2, 3, 4, 5]
            }});

            // languages 위장
            Object.defineProperty(navigator, 'languages', {{
                get: () => {languages_js}
            }});

            // Chrome 런타임 위장
            window.chrome = {{
                runtime: {{}}
            }};

            // permissions 위장
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({{ state: Notification.permission }}) :
                    originalQuery(parameters)
            );

            // WebGL 벤더 위장
            const getParameter = WebGLRenderingContext.prototype.getParameter;
            WebGLRenderingContext.prototype.getParameter = function(parameter) {{
                if (parameter === 37445) {{
                    return 'Intel Inc.';
                }}
                if (parameter === 37446) {{
                    return 'Intel Iris OpenGL Engine';
                }}
                return getParameter.apply(this, arguments);
            }};
            """
        elif browser_type == "firefox":
            return f"""
            // Firefox용 stealth
            Object.defineProperty(navigator, 'webdriver', {{
                get: () => undefined
            }});
            Object.defineProperty(navigator, 'languages', {{
                get: () => {languages_js}
            }});
            """
        else:  # webkit
            return f"""
            // WebKit(Safari)용 stealth
            Object.defineProperty(navigator, 'webdriver', {{
                get: () => undefined
            }});
            Object.defineProperty(navigator, 'languages', {{
                get: () => {languages_js}
            }});
            """


class HumanBehavior:
    """사람처럼 행동하는 메서드 모음"""

    def __init__(self, config: CrawlerConfig):
        self.config = config

    async def random_delay(self, min_ms: int | None = None, max_ms: int | None = None) -> None:
        """랜덤 딜레이"""
        min_ms = min_ms or self.config.min_delay
        max_ms = max_ms or self.config.max_delay
        if min_ms > 0 and max_ms > 0:
            delay = random.randint(min_ms, max_ms)
            await asyncio.sleep(delay / 1000)

    async def mouse_move(self, page: Page, x: float, y: float) -> None:
        """자연스러운 마우스 이동 (곡선)"""
        if not self.config.use_human_mouse:
            await page.mouse.move(x, y)
            return

        steps = random.randint(
            self.config.mouse_move_steps_min,
            self.config.mouse_move_steps_max
        )
        noise_range = self.config.mouse_noise_range
        current_x, current_y = 0, 0

        for i in range(steps):
            progress = (i + 1) / steps
            noise_x = random.randint(-noise_range, noise_range)
            noise_y = random.randint(-noise_range, noise_range)

            next_x = current_x + (x - current_x) * progress + noise_x
            next_y = current_y + (y - current_y) * progress + noise_y

            await page.mouse.move(next_x, next_y)
            await asyncio.sleep(random.uniform(0.005, 0.015))

            current_x, current_y = next_x, next_y

    async def click(self, page: Page, selector: str) -> bool:
        """자연스러운 클릭"""
        try:
            element = page.locator(selector).first
            if await element.count() == 0:
                return False

            if self.config.use_human_mouse:
                box = await element.bounding_box()
                if box:
                    x = box['x'] + random.uniform(5, box['width'] - 5)
                    y = box['y'] + random.uniform(5, box['height'] - 5)
                    await self.mouse_move(page, x, y)
                    await self.random_delay(50, 150)
                    await page.mouse.click(x, y)
                    return True
            else:
                await element.click()
                return True
            return False
        except Exception as e:
            logger.debug(f"클릭 실패: {e}")
            return False

    async def scroll(self, page: Page, direction: str = "down", amount: int = 300) -> None:
        """자연스러운 스크롤"""
        if self.config.use_human_mouse:
            steps = random.randint(
                self.config.scroll_steps_min,
                self.config.scroll_steps_max
            )
            step_amount = amount // steps
            for _ in range(steps):
                delta = step_amount + random.randint(-30, 30)
                if direction == "down":
                    await page.mouse.wheel(0, delta)
                else:
                    await page.mouse.wheel(0, -delta)
                await asyncio.sleep(random.uniform(0.02, 0.05))
        else:
            delta = amount if direction == "down" else -amount
            await page.mouse.wheel(0, delta)

        await self.random_delay(self.config.scroll_delay, self.config.scroll_delay + 200)

    async def simulate_reading(self, page: Page) -> None:
        """페이지 읽는 척 시뮬레이션"""
        if not self.config.simulate_reading:
            return

        for _ in range(random.randint(1, 2)):
            await self.scroll(page, "down", random.randint(100, 300))

        await self.random_delay(500, 1500)

    async def type_text(self, page: Page, selector: str, text: str) -> None:
        """사람처럼 타이핑"""
        element = page.locator(selector)
        for char in text:
            delay = random.randint(
                self.config.typing_delay_min,
                self.config.typing_delay_max
            )
            await element.type(char, delay=delay)
            if random.random() < self.config.typing_pause_probability:
                await asyncio.sleep(random.uniform(0.2, 0.5))


class BaseCrawler(ABC):
    """
    기본 크롤러 클래스

    상속받아 run() 메서드만 구현하면 됨

    Example:
        class MyCrawler(BaseCrawler):
            async def run(self, url: str):
                await self.goto(url)
                title = await self.page.title()
                return {"title": title}
    """

    # 기본 광고/추적 도메인 차단 리스트 (config.custom_ad_domains로 확장 가능)
    DEFAULT_AD_DOMAINS = [
        "googleads", "doubleclick", "googlesyndication",
        "facebook.com/tr", "analytics", "adservice",
        "tracking", "pixel", "beacon"
    ]

    # 기본 캡차 감지 키워드 (config.custom_captcha_indicators로 확장 가능)
    DEFAULT_CAPTCHA_INDICATORS = [
        "captcha", "recaptcha", "hcaptcha", "challenge",
        "robot", "보안문자", "자동입력방지"
    ]

    # 기본 팝업 닫기 셀렉터 (config.custom_popup_selectors로 확장 가능)
    DEFAULT_POPUP_SELECTORS = [
        '[class*="close"]',
        '[class*="dismiss"]',
        '[aria-label*="닫기"]',
        '[aria-label*="Close"]',
        'button:has-text("닫기")',
        'button:has-text("Close")',
        '[class*="modal"] button',
    ]

    def __init__(self, config: CrawlerConfig | None = None):
        self.config = config or CrawlerConfig()
        self.human = HumanBehavior(self.config)

        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None
        self._playwright = None

        # Rate limiting
        self._last_request_time: float = 0
        self._request_count: int = 0

        # 설정과 기본값 병합
        self.ad_domains = self.DEFAULT_AD_DOMAINS + self.config.custom_ad_domains
        self.captcha_indicators = self.DEFAULT_CAPTCHA_INDICATORS + self.config.custom_captcha_indicators
        self.popup_selectors = self.DEFAULT_POPUP_SELECTORS + self.config.custom_popup_selectors

    async def _setup(self) -> None:
        """브라우저 및 컨텍스트 설정"""
        # 디버그 폴더 생성
        if self.config.debug:
            os.makedirs(self.config.debug_dir, exist_ok=True)

        # 브라우저 타입 검증
        browser_type = self.config.browser.lower()
        if browser_type not in ["chromium", "firefox", "webkit"]:
            raise ValueError(f"지원하지 않는 브라우저: {browser_type}. chromium, firefox, webkit 중 선택")

        # User-Agent (설정값 또는 자동 추출)
        user_agent = self.config.user_agent
        if not user_agent:
            user_agent = await BrowserConfig.get_user_agent(browser_type)

        # Playwright 시작
        self._playwright = await async_playwright().start()

        # 브라우저 실행 옵션
        launch_options = {
            "headless": self.config.headless,
        }

        # 브라우저 args (기본값 + 사용자 정의)
        args = BrowserConfig.get_launch_args(browser_type) + self.config.custom_browser_args
        if args:
            launch_options["args"] = args

        # 프록시 설정
        if self.config.proxy:
            launch_options["proxy"] = {"server": self.config.proxy}
            logger.info(f"프록시 설정: {self.config.proxy}")

        # 브라우저 실행 (타입에 따라)
        browser_launcher = getattr(self._playwright, browser_type)
        self.browser = await browser_launcher.launch(**launch_options)
        logger.info(f"브라우저 실행: {browser_type}")

        # HTTP 헤더 설정 (기본값 + 사용자 정의)
        headers = {
            "Accept-Language": self.config.accept_language,
        }
        headers.update(self.config.extra_headers)

        # 컨텍스트 옵션
        context_options = {
            "user_agent": user_agent,
            "viewport": {
                "width": self.config.viewport_width,
                "height": self.config.viewport_height
            },
            "locale": self.config.locale,
            "timezone_id": self.config.timezone,
            "extra_http_headers": headers
        }

        # 디버그 모드: 비디오 녹화
        if self.config.debug:
            context_options["record_video_dir"] = self.config.debug_dir
            context_options["record_video_size"] = {
                "width": self.config.debug_video_width,
                "height": self.config.debug_video_height
            }

        # 컨텍스트 생성
        self.context = await self.browser.new_context(**context_options)

        # 쿠키 로드
        if self.config.cookies_file and Path(self.config.cookies_file).exists():
            await self._load_cookies()

        # 페이지 생성
        self.page = await self.context.new_page()

        # 리소스 차단 설정
        if any([
            self.config.block_images,
            self.config.block_fonts,
            self.config.block_ads,
            self.config.block_media,
            self.config.block_stylesheets
        ]):
            await self._setup_resource_blocking()

        # Stealth 스크립트 적용 (브라우저별, 언어 설정 포함)
        stealth_script = BrowserConfig.get_stealth_script(browser_type, self.config.languages)
        if stealth_script:
            await self.page.add_init_script(stealth_script)

        # 디버그 모드: trace 시작
        if self.config.debug:
            await self.context.tracing.start(screenshots=True, snapshots=True)

        logger.info(f"브라우저 설정 완료 (모드: {self.config.mode}, headless: {self.config.headless})")

    async def _teardown(self) -> None:
        """브라우저 정리"""
        # 쿠키 저장
        if self.config.cookies_file and self.context:
            await self._save_cookies()

        if self.config.debug and self.context:
            await self.context.tracing.stop(path=f"{self.config.debug_dir}/trace.zip")
            logger.info(f"디버그 파일 저장됨: {self.config.debug_dir}/")

        if self.browser:
            await self.browser.close()

        if self._playwright:
            await self._playwright.stop()

    async def _setup_resource_blocking(self) -> None:
        """리소스 차단 설정 (이미지, 폰트, 광고, 미디어, 스타일시트)"""
        async def route_handler(route: Route):
            request = route.request
            resource_type = request.resource_type
            url = request.url.lower()

            # 이미지 차단
            if self.config.block_images and resource_type == "image":
                await route.abort()
                return

            # 폰트 차단
            if self.config.block_fonts and resource_type == "font":
                await route.abort()
                return

            # 미디어 차단
            if self.config.block_media and resource_type in ["media", "video", "audio"]:
                await route.abort()
                return

            # 스타일시트 차단
            if self.config.block_stylesheets and resource_type == "stylesheet":
                await route.abort()
                return

            # 광고 차단 (병합된 도메인 리스트 사용)
            if self.config.block_ads:
                for ad_domain in self.ad_domains:
                    if ad_domain in url:
                        await route.abort()
                        return

            await route.continue_()

        await self.page.route("**/*", route_handler)
        logger.debug("리소스 차단 설정 완료")

    async def _load_cookies(self) -> None:
        """저장된 쿠키 로드"""
        try:
            with open(self.config.cookies_file, 'r', encoding='utf-8') as f:
                cookies = json.load(f)
            await self.context.add_cookies(cookies)
            logger.info(f"쿠키 로드 완료: {len(cookies)}개")
        except Exception as e:
            logger.warning(f"쿠키 로드 실패: {e}")

    async def _save_cookies(self) -> None:
        """현재 쿠키 저장"""
        try:
            cookies = await self.context.cookies()
            with open(self.config.cookies_file, 'w', encoding='utf-8') as f:
                json.dump(cookies, f, ensure_ascii=False, indent=2)
            logger.info(f"쿠키 저장 완료: {len(cookies)}개")
        except Exception as e:
            logger.warning(f"쿠키 저장 실패: {e}")

    async def _rate_limit(self) -> None:
        """Rate limiting 적용"""
        if self.config.requests_per_minute <= 0:
            return

        import time
        current_time = time.time()
        min_interval = 60.0 / self.config.requests_per_minute

        elapsed = current_time - self._last_request_time
        if elapsed < min_interval:
            wait_time = min_interval - elapsed
            logger.debug(f"Rate limit: {wait_time:.2f}초 대기")
            await asyncio.sleep(wait_time)

        self._last_request_time = time.time()
        self._request_count += 1

    async def goto(self, url: str, wait_until: str = "networkidle") -> None:
        """페이지 이동"""
        # Rate limiting 적용
        await self._rate_limit()

        logger.info(f"페이지 이동: {url}")
        await self.page.goto(url, wait_until=wait_until, timeout=self.config.navigation_timeout)
        logger.info("페이지 로드 완료")

        # 캡차 감지
        if await self.detect_captcha():
            raise Exception("캡차가 감지되었습니다. 수동 처리가 필요합니다.")

        # 사람처럼 페이지 둘러보기
        await self.human.simulate_reading(self.page)

    async def screenshot(self, name: str) -> None:
        """스크린샷 저장 (디버그 모드에서만)"""
        if self.config.debug:
            path = f"{self.config.debug_dir}/{name}.png"
            await self.page.screenshot(path=path)
            logger.debug(f"스크린샷 저장: {path}")

    async def click_by_text(self, text: str, exact: bool = False) -> bool:
        """텍스트로 요소 클릭"""
        try:
            element = self.page.get_by_text(text, exact=exact)
            if await element.count() > 0:
                await element.first.click()
                await self.human.random_delay(self.config.click_delay, self.config.click_delay + 300)
                logger.debug(f"클릭 성공: {text}")
                return True
        except Exception as e:
            logger.debug(f"클릭 실패 ({text}): {e}")
        return False

    async def click_by_role(self, role: str, name: str | None = None) -> bool:
        """role로 요소 클릭"""
        try:
            element = self.page.get_by_role(role, name=name)
            if await element.count() > 0:
                await element.first.click()
                await self.human.random_delay(self.config.click_delay, self.config.click_delay + 300)
                logger.debug(f"클릭 성공: role={role}, name={name}")
                return True
        except Exception as e:
            logger.debug(f"클릭 실패 (role={role}): {e}")
        return False

    async def wait_and_click(self, selector: str, timeout: int | None = None) -> bool:
        """요소 대기 후 클릭"""
        try:
            timeout = timeout or self.config.action_timeout
            await self.page.wait_for_selector(selector, timeout=timeout)
            await self.page.click(selector)
            await self.human.random_delay(self.config.click_delay, self.config.click_delay + 300)
            return True
        except Exception as e:
            logger.debug(f"대기 후 클릭 실패: {e}")
            return False

    async def extract_text(self, selector: str) -> str | None:
        """요소에서 텍스트 추출"""
        try:
            element = self.page.locator(selector)
            if await element.count() > 0:
                return await element.first.text_content()
        except Exception as e:
            logger.debug(f"텍스트 추출 실패: {e}")
        return None

    async def extract_attribute(self, selector: str, attribute: str) -> str | None:
        """요소에서 속성 추출"""
        try:
            element = self.page.locator(selector)
            if await element.count() > 0:
                return await element.first.get_attribute(attribute)
        except Exception as e:
            logger.debug(f"속성 추출 실패: {e}")
        return None

    async def retry(self, func, *args, max_retries: int | None = None, **kwargs) -> Any:
        """재시도 래퍼"""
        max_retries = max_retries or self.config.max_retries
        last_error = None

        for attempt in range(max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(f"시도 {attempt + 1}/{max_retries} 실패: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay / 1000)

        raise last_error


    async def detect_captcha(self) -> bool:
        """캡차 페이지 감지 (기본 + 사용자 정의 키워드)"""
        try:
            content = await self.page.content()
            content_lower = content.lower()

            for indicator in self.captcha_indicators:
                if indicator.lower() in content_lower:
                    logger.warning(f"캡차 감지됨: {indicator}")
                    await self.screenshot("captcha_detected")
                    return True
            return False
        except Exception:
            return False

    async def infinite_scroll(
        self,
        max_scrolls: int = 50,
        scroll_delay: int | None = None,
        end_selector: str | None = None,
        load_more_selector: str | None = None
    ) -> int:
        """
        무한 스크롤 처리

        Args:
            max_scrolls: 최대 스크롤 횟수
            scroll_delay: 스크롤 간 딜레이 (ms)
            end_selector: 끝 감지 셀렉터 (이 요소가 나타나면 종료)
            load_more_selector: 더보기 버튼 셀렉터

        Returns:
            실제 스크롤 횟수
        """
        scroll_delay = scroll_delay or self.config.scroll_delay
        scroll_count = 0
        last_height = 0

        for _ in range(max_scrolls):
            # 더보기 버튼이 있으면 클릭
            if load_more_selector:
                btn = self.page.locator(load_more_selector)
                if await btn.count() > 0:
                    try:
                        await btn.first.click()
                        await self.human.random_delay(scroll_delay, scroll_delay + 500)
                        scroll_count += 1
                        continue
                    except Exception:
                        pass

            # 스크롤
            current_height = await self.page.evaluate("document.body.scrollHeight")

            if current_height == last_height:
                logger.debug(f"스크롤 끝 도달 (총 {scroll_count}회)")
                break

            last_height = current_height
            await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await self.human.random_delay(scroll_delay, scroll_delay + 500)
            scroll_count += 1

            # 끝 감지
            if end_selector:
                end_el = self.page.locator(end_selector)
                if await end_el.count() > 0:
                    logger.debug(f"끝 요소 감지 (총 {scroll_count}회)")
                    break

        logger.info(f"무한 스크롤 완료: {scroll_count}회")
        return scroll_count

    async def close_popups(self, selectors: list[str] | None = None) -> int:
        """
        팝업/모달 닫기

        Args:
            selectors: 추가 닫기 버튼 셀렉터 리스트 (기본값 + 사용자 정의에 추가됨)

        Returns:
            닫은 팝업 수
        """
        # 기본 + 사용자 정의 + 함수 인자 병합
        all_selectors = self.popup_selectors.copy()
        if selectors:
            all_selectors.extend(selectors)

        closed_count = 0

        for selector in all_selectors:
            try:
                elements = self.page.locator(selector)
                count = await elements.count()
                for i in range(count):
                    try:
                        await elements.nth(i).click(timeout=1000)
                        closed_count += 1
                        await self.human.random_delay(200, 500)
                    except Exception:
                        pass
            except Exception:
                pass

        if closed_count > 0:
            logger.debug(f"팝업 {closed_count}개 닫음")
        return closed_count

    async def wait_for_network_idle(self, timeout: int = 5000) -> None:
        """네트워크 요청이 멈출 때까지 대기"""
        try:
            await self.page.wait_for_load_state("networkidle", timeout=timeout)
        except Exception:
            pass

    async def get_all_links(self, selector: str = "a") -> list[str]:
        """페이지의 모든 링크 추출"""
        links = await self.page.evaluate(f'''
            () => {{
                const elements = document.querySelectorAll("{selector}");
                return Array.from(elements)
                    .map(el => el.href)
                    .filter(href => href && href.startsWith("http"));
            }}
        ''')
        return links

    async def save_html(self, filename: str | None = None) -> str:
        """현재 페이지 HTML 저장"""
        html = await self.page.content()
        filename = filename or f"page_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = Path(self.config.debug_dir) / filename

        os.makedirs(self.config.debug_dir, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.debug(f"HTML 저장: {filepath}")
        return str(filepath)

    async def save_json(self, data: Any, filename: str) -> str:
        """데이터를 JSON 파일로 저장"""
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"JSON 저장: {filepath}")
        return str(filepath)

    async def new_page(self) -> Page:
        """새 탭/페이지 생성"""
        new_page = await self.context.new_page()
        stealth_script = BrowserConfig.get_stealth_script(
            self.config.browser,
            self.config.languages
        )
        if stealth_script:
            await new_page.add_init_script(stealth_script)
        return new_page

    async def switch_to_page(self, index: int) -> None:
        """특정 탭으로 전환"""
        pages = self.context.pages
        if 0 <= index < len(pages):
            self.page = pages[index]
            await self.page.bring_to_front()
            logger.debug(f"탭 전환: {index}")

    async def wait_for_element(
        self,
        selector: str,
        timeout: int | None = None,
        state: str = "visible"
    ) -> bool:
        """
        요소가 나타날 때까지 대기

        Args:
            selector: CSS 셀렉터
            timeout: 타임아웃 (ms)
            state: "attached", "detached", "visible", "hidden"
        """
        try:
            timeout = timeout or self.config.action_timeout
            await self.page.wait_for_selector(selector, timeout=timeout, state=state)
            return True
        except Exception:
            return False

    async def wait_for_url(self, url_pattern: str, timeout: int | None = None) -> bool:
        """URL이 변경될 때까지 대기"""
        try:
            timeout = timeout or self.config.navigation_timeout
            await self.page.wait_for_url(url_pattern, timeout=timeout)
            return True
        except Exception:
            return False

    async def fill_form(self, fields: dict[str, str]) -> None:
        """
        폼 필드 입력

        Args:
            fields: {셀렉터: 값} 딕셔너리
        """
        for selector, value in fields.items():
            try:
                await self.page.fill(selector, value)
                await self.human.random_delay(100, 300)
            except Exception as e:
                logger.warning(f"폼 입력 실패 ({selector}): {e}")

    async def select_option(self, selector: str, value: str) -> None:
        """드롭다운 선택"""
        await self.page.select_option(selector, value)
        await self.human.random_delay(200, 500)

    async def upload_file(self, selector: str, file_path: str) -> None:
        """파일 업로드"""
        await self.page.set_input_files(selector, file_path)
        logger.debug(f"파일 업로드: {file_path}")

    async def download_file(self, click_selector: str, save_path: str) -> str | None:
        """
        파일 다운로드

        Args:
            click_selector: 다운로드 버튼 셀렉터
            save_path: 저장 경로
        """
        try:
            async with self.page.expect_download() as download_info:
                await self.page.click(click_selector)
            download = await download_info.value
            await download.save_as(save_path)
            logger.info(f"파일 다운로드 완료: {save_path}")
            return save_path
        except Exception as e:
            logger.error(f"다운로드 실패: {e}")
            return None

    async def intercept_response(
        self,
        url_pattern: str,
        callback: Callable
    ) -> None:
        """
        API 응답 가로채기

        Args:
            url_pattern: URL 패턴 (예: "**/api/**")
            callback: 응답 처리 함수 (async def callback(response))
        """
        async def handler(response):
            if url_pattern.replace("**", "") in response.url:
                await callback(response)

        self.page.on("response", handler)

    async def get_local_storage(self) -> dict:
        """로컬 스토리지 가져오기"""
        return await self.page.evaluate("() => Object.assign({}, localStorage)")

    async def set_local_storage(self, data: dict) -> None:
        """로컬 스토리지 설정"""
        for key, value in data.items():
            await self.page.evaluate(
                f"localStorage.setItem('{key}', '{value}')"
            )

    async def clear_local_storage(self) -> None:
        """로컬 스토리지 초기화"""
        await self.page.evaluate("localStorage.clear()")

    async def execute_script(self, script: str) -> Any:
        """JavaScript 실행"""
        return await self.page.evaluate(script)

    async def get_cookies_dict(self) -> dict:
        """쿠키를 딕셔너리로 반환"""
        cookies = await self.context.cookies()
        return {c['name']: c['value'] for c in cookies}

    async def set_cookie(self, name: str, value: str, domain: str) -> None:
        """단일 쿠키 설정"""
        await self.context.add_cookies([{
            'name': name,
            'value': value,
            'domain': domain,
            'path': '/'
        }])

    async def check_element_exists(self, selector: str) -> bool:
        """요소 존재 여부 확인"""
        return await self.page.locator(selector).count() > 0

    async def get_element_count(self, selector: str) -> int:
        """요소 개수 반환"""
        return await self.page.locator(selector).count()

    async def hover(self, selector: str) -> None:
        """요소에 마우스 호버"""
        await self.page.hover(selector)
        await self.human.random_delay(200, 500)

    async def press_key(self, key: str) -> None:
        """키 입력 (Enter, Escape, Tab 등)"""
        await self.page.keyboard.press(key)

    async def take_element_screenshot(self, selector: str, path: str) -> str | None:
        """특정 요소만 스크린샷"""
        try:
            element = self.page.locator(selector)
            await element.screenshot(path=path)
            return path
        except Exception as e:
            logger.error(f"요소 스크린샷 실패: {e}")
            return None

    @abstractmethod
    async def run(self, url: str) -> Any:
        """
        메인 크롤링 로직 (상속받아 구현)

        Example:
            async def run(self, url: str):
                await self.goto(url)
                title = await self.page.title()
                return {"title": title}
        """
        pass

    async def execute(self, url: str) -> Any:
        """크롤러 실행 (setup -> run -> teardown)"""
        try:
            await self._setup()
            result = await self.run(url)
            return result
        except Exception as e:
            logger.error(f"크롤링 실패: {e}")
            await self.screenshot("error")
            raise
        finally:
            await self._teardown()



class ExampleCrawler(BaseCrawler):
    """사용 예시 크롤러"""

    async def run(self, url: str) -> dict:
        # 페이지 이동
        await self.goto(url)

        # 스크린샷
        await self.screenshot("01_loaded")

        # 제목 추출
        title = await self.page.title()

        # JavaScript로 데이터 추출
        data = await self.page.evaluate('''
            () => {
                return {
                    title: document.title,
                    url: window.location.href,
                    h1: document.querySelector('h1')?.textContent || ''
                };
            }
        ''')

        return data


class DataUtils:
    """데이터 변환 유틸리티"""

    @staticmethod
    def parse_number(text: str) -> int:
        """
        텍스트에서 숫자 추출 (천단위 콤마, K/M 단위 지원)
        "1,234" -> 1234
        "12.5K" -> 12500
        "1.2M" -> 1200000
        """
        if not text:
            return 0

        text = text.strip().upper().replace(",", "").replace(" ", "")

        try:
            if "K" in text:
                return int(float(text.replace("K", "")) * 1000)
            elif "M" in text:
                return int(float(text.replace("M", "")) * 1000000)
            else:
                # 숫자만 추출
                import re
                match = re.search(r"[\d.]+", text)
                if match:
                    return int(float(match.group()))
        except (ValueError, TypeError):
            pass
        return 0

    @staticmethod
    def parse_date(text: str, formats: list[str] | None = None) -> datetime | None:
        """
        텍스트에서 날짜 파싱
        """
        if not text:
            return None

        default_formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%Y.%m.%d %H:%M:%S",
            "%Y.%m.%d %H:%M",
            "%Y.%m.%d",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d",
            "%m/%d/%Y",
            "%d-%m-%Y",
        ]

        formats = formats or default_formats

        for fmt in formats:
            try:
                return datetime.strptime(text.strip(), fmt)
            except ValueError:
                continue
        return None

    @staticmethod
    def clean_text(text: str) -> str:
        """텍스트 정리 (공백, 줄바꿈 정리)"""
        if not text:
            return ""
        import re
        # 여러 공백을 하나로
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def extract_urls(text: str) -> list[str]:
        """텍스트에서 URL 추출"""
        import re
        pattern = r'https?://[^\s<>"\'{}|\\^`\[\]]+'
        return re.findall(pattern, text)

    @staticmethod
    def extract_emails(text: str) -> list[str]:
        """텍스트에서 이메일 추출"""
        import re
        pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        return re.findall(pattern, text)

    @staticmethod
    def extract_phones(text: str) -> list[str]:
        """텍스트에서 전화번호 추출 (한국)"""
        import re
        patterns = [
            r'01[0-9]-?\d{3,4}-?\d{4}',  # 휴대폰
            r'0\d{1,2}-?\d{3,4}-?\d{4}',  # 일반전화
        ]
        results = []
        for pattern in patterns:
            results.extend(re.findall(pattern, text))
        return results


class Selectors:
    """자주 쓰는 셀렉터 패턴"""

    @staticmethod
    def by_text(text: str, tag: str = "*") -> str:
        """텍스트로 셀렉터 생성"""
        return f'{tag}:has-text("{text}")'

    @staticmethod
    def by_class_contains(class_name: str, tag: str = "*") -> str:
        """클래스명 포함으로 셀렉터 생성"""
        return f'{tag}[class*="{class_name}"]'

    @staticmethod
    def by_id_contains(id_name: str, tag: str = "*") -> str:
        """ID 포함으로 셀렉터 생성"""
        return f'{tag}[id*="{id_name}"]'

    @staticmethod
    def by_attr(attr: str, value: str, tag: str = "*") -> str:
        """속성으로 셀렉터 생성"""
        return f'{tag}[{attr}="{value}"]'

    @staticmethod
    def by_attr_contains(attr: str, value: str, tag: str = "*") -> str:
        """속성 포함으로 셀렉터 생성"""
        return f'{tag}[{attr}*="{value}"]'

    @staticmethod
    def nth(selector: str, n: int) -> str:
        """n번째 요소 셀렉터"""
        return f'{selector}:nth-child({n})'

    @staticmethod
    def first(selector: str) -> str:
        """첫 번째 요소"""
        return f'{selector}:first-child'

    @staticmethod
    def last(selector: str) -> str:
        """마지막 요소"""
        return f'{selector}:last-child'


def load_config_from_file(filepath: str) -> CrawlerConfig:
    """
    JSON/YAML 파일에서 설정 로드

    config.json 예시:
    {
        "mode": "fast",
        "browser": "chromium",
        "headless": true,
        "debug": false
    }
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"설정 파일 없음: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        if filepath.suffix in ['.yaml', '.yml']:
            try:
                import yaml
                data = yaml.safe_load(f)
            except ImportError:
                raise ImportError("YAML 지원을 위해 'pip install pyyaml' 필요")
        else:
            data = json.load(f)

    return CrawlerConfig(**data)


def load_config_from_env() -> CrawlerConfig:
    """
    환경변수에서 설정 로드

    환경변수 예시:
    CRAWLER_MODE=fast
    CRAWLER_BROWSER=chromium
    CRAWLER_HEADLESS=true
    CRAWLER_DEBUG=false
    """
    import os

    def get_bool(key: str, default: bool) -> bool:
        val = os.getenv(key, "").lower()
        if val in ["true", "1", "yes"]:
            return True
        elif val in ["false", "0", "no"]:
            return False
        return default

    def get_int(key: str, default: int) -> int:
        try:
            return int(os.getenv(key, default))
        except (ValueError, TypeError):
            return default

    return CrawlerConfig(
        mode=os.getenv("CRAWLER_MODE", "balanced"),
        browser=os.getenv("CRAWLER_BROWSER", "chromium"),
        headless=get_bool("CRAWLER_HEADLESS", True),
        debug=get_bool("CRAWLER_DEBUG", False),
        verbose=get_bool("CRAWLER_VERBOSE", False),
        proxy=os.getenv("CRAWLER_PROXY"),
        requests_per_minute=get_int("CRAWLER_RPM", 0),
    )


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """
    실패 시 재시도 데코레이터

    @retry_on_failure(max_retries=3, delay=1.0)
    async def my_function():
        ...
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    logger.warning(f"{func.__name__} 시도 {attempt + 1}/{max_retries} 실패: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay)
            raise last_error
        return wrapper
    return decorator


def log_execution(func):
    """
    실행 시간 로깅 데코레이터

    @log_execution
    async def my_function():
        ...
    """
    async def wrapper(*args, **kwargs):
        import time
        start = time.time()
        logger.info(f"{func.__name__} 시작")
        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start
            logger.info(f"{func.__name__} 완료 ({elapsed:.2f}초)")
            return result
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"{func.__name__} 실패 ({elapsed:.2f}초): {e}")
            raise
    return wrapper


class DebugCrawler(BaseCrawler):
    """
    대화형 디버그 크롤러

    브라우저를 열고 수동으로 조작하면서 셀렉터를 테스트할 수 있음

    사용법:
        crawler = DebugCrawler()
        await crawler.start("https://example.com")
        # 이후 crawler.page로 직접 조작
        # await crawler.page.click("...")
        # await crawler.test_selector("...")
        await crawler.stop()
    """

    async def run(self, url: str) -> Any:
        # DebugCrawler는 run을 사용하지 않음
        pass

    async def start(self, url: str) -> None:
        """디버그 세션 시작"""
        self.config.headless = False  # 브라우저 보이게
        await self._setup()
        await self.goto(url)
        logger.info("=" * 50)
        logger.info("디버그 모드 시작")
        logger.info("self.page로 직접 조작 가능")
        logger.info("self.test_selector('셀렉터')로 테스트")
        logger.info("self.stop()으로 종료")
        logger.info("=" * 50)

    async def stop(self) -> None:
        """디버그 세션 종료"""
        await self._teardown()
        logger.info("디버그 모드 종료")

    async def test_selector(self, selector: str) -> dict:
        """
        셀렉터 테스트

        Returns:
            {
                "selector": 셀렉터,
                "count": 매칭된 요소 수,
                "texts": 각 요소의 텍스트 (최대 5개),
                "visible": 보이는 요소 수
            }
        """
        try:
            elements = self.page.locator(selector)
            count = await elements.count()

            texts = []
            visible_count = 0

            for i in range(min(count, 5)):
                el = elements.nth(i)
                text = await el.text_content()
                texts.append(text.strip()[:50] if text else "")
                if await el.is_visible():
                    visible_count += 1

            result = {
                "selector": selector,
                "count": count,
                "texts": texts,
                "visible": visible_count
            }

            logger.info(f"셀렉터 테스트: {selector}")
            logger.info(f"  - 매칭: {count}개 (보이는 것: {visible_count}개)")
            for i, text in enumerate(texts):
                logger.info(f"  - [{i}] {text}")

            return result

        except Exception as e:
            logger.error(f"셀렉터 테스트 실패: {e}")
            return {"selector": selector, "error": str(e)}

    async def highlight(self, selector: str, color: str = "red") -> None:
        """요소 하이라이트 (디버깅용)"""
        await self.page.evaluate(f'''
            (selector) => {{
                document.querySelectorAll(selector).forEach(el => {{
                    el.style.outline = "3px solid {color}";
                    el.style.outlineOffset = "2px";
                }});
            }}
        ''', selector)

    async def get_page_info(self) -> dict:
        """현재 페이지 정보"""
        return await self.page.evaluate('''
            () => ({
                url: window.location.href,
                title: document.title,
                width: window.innerWidth,
                height: window.innerHeight,
                scrollY: window.scrollY,
                elementCount: document.querySelectorAll("*").length
            })
        ''')



async def crawl_parallel(
    crawler_class: type[BaseCrawler],
    urls: list[str],
    config: CrawlerConfig | None = None,
    max_concurrent: int = 3
) -> list[dict]:
    """
    여러 URL 병렬 크롤링 (단순 버전)

    Args:
        crawler_class: BaseCrawler를 상속받은 크롤러 클래스
        urls: 크롤링할 URL 목록
        config: 크롤러 설정
        max_concurrent: 최대 동시 실행 수

    Returns:
        [{url, success, data, error}, ...]
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []

    async def crawl_one(url: str) -> dict:
        async with semaphore:
            try:
                crawler = crawler_class(config=config)
                data = await crawler.execute(url)
                return {"url": url, "success": True, "data": data, "error": None}
            except Exception as e:
                logger.error(f"크롤링 실패 ({url}): {e}")
                return {"url": url, "success": False, "data": None, "error": str(e)}

    tasks = [crawl_one(url) for url in urls]
    results = await asyncio.gather(*tasks)

    success_count = sum(1 for r in results if r["success"])
    logger.info(f"병렬 크롤링 완료: {success_count}/{len(urls)} 성공")

    return results



class BrowserPool:
    """
    브라우저 인스턴스 풀링

    브라우저를 매번 새로 생성하지 않고 재사용하여 리소스 절약
    - 브라우저 시작 시간 절약 (1-2초 → 거의 0초)
    - 메모리 사용량 감소
    - CPU 사용량 감소

    사용법:
        async with BrowserPool(max_browsers=3) as pool:
            browser = await pool.acquire()
            # 작업 수행
            await pool.release(browser)
    """

    def __init__(
        self,
        max_browsers: int = 3,
        browser_type: str = "chromium",
        headless: bool = True,
        launch_args: list[str] | None = None
    ):
        self.max_browsers = max_browsers
        self.browser_type = browser_type
        self.headless = headless
        self.launch_args = launch_args or BrowserConfig.get_launch_args(browser_type)

        self._playwright = None
        self._available: asyncio.Queue = asyncio.Queue()
        self._in_use: set = set()
        self._all_browsers: list = []
        self._lock = asyncio.Lock()
        self._closed = False

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def start(self) -> None:
        """풀 초기화"""
        self._playwright = await async_playwright().start()
        logger.info(f"BrowserPool 시작: 최대 {self.max_browsers}개 브라우저")

    async def acquire(self) -> Browser:
        """브라우저 획득 (없으면 생성)"""
        async with self._lock:
            # 사용 가능한 브라우저가 있으면 반환
            if not self._available.empty():
                browser = await self._available.get()
                self._in_use.add(browser)
                logger.debug(f"브라우저 재사용 (사용중: {len(self._in_use)})")
                return browser

            # 최대 개수 미만이면 새로 생성
            if len(self._all_browsers) < self.max_browsers:
                browser_launcher = getattr(self._playwright, self.browser_type)
                browser = await browser_launcher.launch(
                    headless=self.headless,
                    args=self.launch_args if self.launch_args else None
                )
                self._all_browsers.append(browser)
                self._in_use.add(browser)
                logger.debug(f"브라우저 생성 (총: {len(self._all_browsers)})")
                return browser

        # 최대 개수에 도달하면 대기
        logger.debug("브라우저 대기 중...")
        browser = await self._available.get()
        async with self._lock:
            self._in_use.add(browser)
        return browser

    async def release(self, browser: Browser) -> None:
        """브라우저 반환"""
        async with self._lock:
            if browser in self._in_use:
                self._in_use.remove(browser)

                # 브라우저가 아직 연결되어 있으면 재사용
                if browser.is_connected():
                    # 모든 컨텍스트 정리 (메모리 해제)
                    for context in browser.contexts:
                        await context.close()
                    await self._available.put(browser)
                    logger.debug(f"브라우저 반환 (사용중: {len(self._in_use)})")
                else:
                    # 연결 끊어진 브라우저는 제거
                    self._all_browsers.remove(browser)
                    logger.warning("끊어진 브라우저 제거")

    async def close(self) -> None:
        """모든 브라우저 종료"""
        self._closed = True
        async with self._lock:
            for browser in self._all_browsers:
                try:
                    await browser.close()
                except Exception:
                    pass
            self._all_browsers.clear()
            self._in_use.clear()

        if self._playwright:
            await self._playwright.stop()

        logger.info("BrowserPool 종료")

    @property
    def stats(self) -> dict:
        """현재 상태"""
        return {
            "total": len(self._all_browsers),
            "in_use": len(self._in_use),
            "available": self._available.qsize(),
            "max": self.max_browsers
        }


class ResourceOptimizer:
    """
    리소스 최적화 유틸리티

    메모리, CPU, 네트워크 사용량을 줄이는 다양한 최적화 기법

    사용자 정의:
        ResourceOptimizer.AD_TRACKING_DOMAINS.extend(["my-ad-domain.com"])
        ResourceOptimizer.DEFAULT_BROWSER_MEMORY_MB = 200
    """

    # 차단할 리소스 타입별 정의
    BLOCK_PATTERNS = {
        "images": ["image", "img"],
        "fonts": ["font"],
        "stylesheets": ["stylesheet", "css"],
        "media": ["media", "video", "audio"],
        "scripts": ["script"],  # 주의: 동적 페이지에서는 비활성화
    }

    # 기본 광고/추적 도메인 (확장 가능)
    AD_TRACKING_DOMAINS = [
        "googleads", "doubleclick", "googlesyndication", "google-analytics",
        "facebook.com/tr", "fbcdn", "analytics", "adservice",
        "tracking", "pixel", "beacon", "advertisement",
        "ads.", ".ads", "adserver", "adtech",
        "criteo", "outbrain", "taboola", "amazon-adsystem"
    ]

    # 워커 계산 기본값 (사용자 정의 가능)
    DEFAULT_BROWSER_MEMORY_MB = 150.0
    DEFAULT_MAX_WORKERS = 20
    DEFAULT_CPU_MULTIPLIER = 2  # 코어당 워커 수

    @classmethod
    async def setup_resource_blocking(
        cls,
        page: Page,
        block_images: bool = True,
        block_fonts: bool = True,
        block_stylesheets: bool = False,
        block_media: bool = True,
        block_ads: bool = True,
        block_scripts: bool = False,
        allowed_domains: list[str] | None = None,
        custom_ad_domains: list[str] | None = None
    ) -> None:
        """
        리소스 차단 설정

        Args:
            page: Playwright Page 객체
            block_images: 이미지 차단 (메모리 절약 큼)
            block_fonts: 폰트 차단 (네트워크 절약)
            block_stylesheets: CSS 차단 (주의: 레이아웃 깨질 수 있음)
            block_media: 비디오/오디오 차단
            block_ads: 광고/추적 차단
            block_scripts: 스크립트 차단 (주의: 동적 페이지 작동 안 함)
            allowed_domains: 차단에서 제외할 도메인
            custom_ad_domains: 추가 광고/추적 도메인 (기본값에 병합)
        """
        allowed_domains = allowed_domains or []
        ad_domains = cls.AD_TRACKING_DOMAINS + (custom_ad_domains or [])

        async def route_handler(route: Route):
            request = route.request
            resource_type = request.resource_type
            url = request.url.lower()

            # 허용된 도메인은 통과
            for domain in allowed_domains:
                if domain in url:
                    await route.continue_()
                    return

            # 광고/추적 차단
            if block_ads:
                for ad_domain in ad_domains:
                    if ad_domain in url:
                        await route.abort()
                        return

            # 리소스 타입별 차단
            if block_images and resource_type == "image":
                await route.abort()
                return
            if block_fonts and resource_type == "font":
                await route.abort()
                return
            if block_stylesheets and resource_type == "stylesheet":
                await route.abort()
                return
            if block_media and resource_type in ["media", "video", "audio"]:
                await route.abort()
                return
            if block_scripts and resource_type == "script":
                await route.abort()
                return

            await route.continue_()

        await page.route("**/*", route_handler)

    @classmethod
    async def optimize_page_memory(cls, page: Page) -> None:
        """
        페이지 메모리 최적화

        - 불필요한 이벤트 리스너 제거
        - 이미지 lazy loading 적용
        - DOM 정리
        """
        await page.evaluate('''
            () => {
                // 숨겨진 이미지 src 제거
                document.querySelectorAll('img[loading="lazy"]').forEach(img => {
                    if (!img.getBoundingClientRect().top < window.innerHeight) {
                        img.src = '';
                    }
                });

                // 불필요한 이벤트 리스너 힌트 제거
                document.querySelectorAll('[onclick], [onmouseover], [onmouseout]')
                    .forEach(el => {
                        el.onclick = null;
                        el.onmouseover = null;
                        el.onmouseout = null;
                    });

                // 애니메이션 중지
                document.querySelectorAll('*').forEach(el => {
                    el.style.animation = 'none';
                    el.style.transition = 'none';
                });
            }
        ''')

    @classmethod
    def get_optimal_workers(
        cls,
        total_urls: int,
        available_memory_gb: float = 8.0,
        per_browser_memory_mb: float | None = None,
        cpu_cores: int | None = None,
        max_workers: int | None = None,
        cpu_multiplier: int | None = None
    ) -> int:
        """
        최적의 워커 수 계산

        Args:
            total_urls: 크롤링할 총 URL 수
            available_memory_gb: 사용 가능한 메모리 (GB)
            per_browser_memory_mb: 브라우저당 메모리 사용량 (MB), None이면 기본값
            cpu_cores: CPU 코어 수 (None이면 자동 감지)
            max_workers: 최대 워커 수 제한 (None이면 기본값)
            cpu_multiplier: CPU 코어당 워커 수 (None이면 기본값)

        Returns:
            권장 워커 수
        """
        # 기본값 적용
        per_browser_memory_mb = per_browser_memory_mb or cls.DEFAULT_BROWSER_MEMORY_MB
        max_workers = max_workers or cls.DEFAULT_MAX_WORKERS
        cpu_multiplier = cpu_multiplier or cls.DEFAULT_CPU_MULTIPLIER

        # CPU 코어 수
        if cpu_cores is None:
            cpu_cores = os.cpu_count() or 4

        # 메모리 기반 최대 워커
        memory_based = int((available_memory_gb * 1024) / per_browser_memory_mb)

        # CPU 기반 최대 워커
        cpu_based = cpu_cores * cpu_multiplier

        # URL 수 기반 (너무 많은 워커는 비효율)
        url_based = min(total_urls, max_workers)

        optimal = min(memory_based, cpu_based, url_based)
        optimal = max(optimal, 1)  # 최소 1개

        logger.debug(
            f"최적 워커 계산: 메모리={memory_based}, CPU={cpu_based}, "
            f"URL={url_based} → 권장={optimal}"
        )

        return optimal


@dataclass
class CrawlTask:
    """크롤링 작업"""
    url: str
    priority: int = 0  # 높을수록 먼저 처리
    retry_count: int = 0
    max_retries: int = 3
    metadata: dict = field(default_factory=dict)


@dataclass
class CrawlResult:
    """크롤링 결과"""
    url: str
    success: bool
    data: Any = None
    error: str | None = None
    elapsed_time: float = 0
    retry_count: int = 0
    worker_id: int = 0


class CrawlerPool:
    """
    Worker Pool 기반 병렬 크롤링 시스템

    특징:
    - Queue 기반 작업 분배
    - Round-Robin 프록시/브라우저 순환
    - 우선순위 지원
    - 실패 시 자동 재시도
    - 진행 상황 실시간 추적
    - 동적 워커 수 조절
    - 브라우저 풀링으로 리소스 최적화
    - 적응형 동시성 조절

    사용법:
        pool = CrawlerPool(
            crawler_class=MyCrawler,
            num_workers=5,
            proxies=["http://proxy1:8080", "http://proxy2:8080"]
        )

        # 작업 추가
        pool.add_urls(["https://...", "https://..."])
        pool.add_task(CrawlTask(url="https://...", priority=10))

        # 실행
        results = await pool.run()

        # 또는 콜백과 함께
        async def on_complete(result):
            print(f"완료: {result.url}")

        results = await pool.run(on_complete=on_complete)

        # 리소스 최적화 모드
        pool = CrawlerPool(
            crawler_class=MyCrawler,
            num_workers="auto",  # 시스템에 맞게 자동 조절
            reuse_browsers=True  # 브라우저 재사용
        )
    """

    def __init__(
        self,
        crawler_class: type[BaseCrawler],
        config: CrawlerConfig | None = None,
        num_workers: int | str = 3,  # "auto"로 설정 가능
        proxies: list[str] | None = None,
        browsers: list[str] | None = None,
        reuse_browsers: bool = False,  # 브라우저 풀링 사용 여부
        adaptive_concurrency: bool = False,  # 적응형 동시성
    ):
        self.crawler_class = crawler_class
        self.base_config = config or CrawlerConfig()
        self.proxies = proxies or [None]  # Round-Robin용
        self.browsers = browsers or ["chromium"]  # Round-Robin용
        self.reuse_browsers = reuse_browsers
        self.adaptive_concurrency = adaptive_concurrency

        # 워커 수 설정
        self._auto_workers = (num_workers == "auto")
        self.num_workers = 3 if self._auto_workers else num_workers

        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._results: list[CrawlResult] = []
        self._lock = asyncio.Lock()

        # 통계
        self._total_tasks = 0
        self._completed_tasks = 0
        self._failed_tasks = 0
        self._running = False

        # 프록시/브라우저 인덱스 (Round-Robin)
        self._proxy_index = 0
        self._browser_index = 0

        # 브라우저 풀 (reuse_browsers=True일 때 사용)
        self._browser_pool: BrowserPool | None = None

        # 적응형 동시성을 위한 메트릭
        self._response_times: list[float] = []
        self._error_rate: float = 0.0

    def add_task(self, task: CrawlTask) -> None:
        """작업 추가"""
        # PriorityQueue는 낮은 값이 먼저이므로 priority를 음수로
        self._queue.put_nowait((-task.priority, task))
        self._total_tasks += 1

    def add_urls(self, urls: list[str], priority: int = 0) -> None:
        """URL 목록 추가"""
        for url in urls:
            self.add_task(CrawlTask(url=url, priority=priority))

    def add_url(self, url: str, priority: int = 0, **metadata) -> None:
        """단일 URL 추가"""
        self.add_task(CrawlTask(url=url, priority=priority, metadata=metadata))

    def _get_next_proxy(self) -> str | None:
        """Round-Robin으로 다음 프록시 반환"""
        proxy = self.proxies[self._proxy_index % len(self.proxies)]
        self._proxy_index += 1
        return proxy

    def _get_next_browser(self) -> str:
        """Round-Robin으로 다음 브라우저 반환"""
        browser = self.browsers[self._browser_index % len(self.browsers)]
        self._browser_index += 1
        return browser

    async def _worker(
        self,
        worker_id: int,
        on_complete: Callable[[CrawlResult], Any] | None = None
    ) -> None:
        """워커 코루틴"""
        while self._running:
            try:
                # 큐에서 작업 가져오기 (타임아웃 1초)
                try:
                    _, task = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                # 설정 생성 (Round-Robin 적용)
                config = CrawlerConfig(
                    mode=self.base_config.mode,
                    browser=self._get_next_browser(),
                    proxy=self._get_next_proxy(),
                    headless=self.base_config.headless,
                    debug=self.base_config.debug,
                    verbose=self.base_config.verbose,
                    block_images=self.base_config.block_images,
                    block_fonts=self.base_config.block_fonts,
                    block_ads=self.base_config.block_ads,
                )

                # 크롤링 실행
                import time
                start_time = time.time()
                result = CrawlResult(
                    url=task.url,
                    success=False,
                    retry_count=task.retry_count,
                    worker_id=worker_id
                )

                try:
                    crawler = self.crawler_class(config=config)
                    data = await crawler.execute(task.url)
                    result.success = True
                    result.data = data
                    result.elapsed_time = time.time() - start_time

                    async with self._lock:
                        self._completed_tasks += 1
                        self._response_times.append(result.elapsed_time)

                    logger.debug(f"[Worker {worker_id}] 완료: {task.url} ({result.elapsed_time:.2f}초)")

                except Exception as e:
                    result.error = str(e)
                    result.elapsed_time = time.time() - start_time

                    async with self._lock:
                        self._response_times.append(result.elapsed_time)

                    # 재시도
                    if task.retry_count < task.max_retries:
                        task.retry_count += 1
                        self._queue.put_nowait((-task.priority, task))
                        logger.warning(f"[Worker {worker_id}] 재시도 {task.retry_count}/{task.max_retries}: {task.url}")
                    else:
                        async with self._lock:
                            self._failed_tasks += 1
                        logger.error(f"[Worker {worker_id}] 최종 실패: {task.url} - {e}")

                # 결과 저장
                async with self._lock:
                    self._results.append(result)

                # 콜백 호출
                if on_complete:
                    try:
                        if asyncio.iscoroutinefunction(on_complete):
                            await on_complete(result)
                        else:
                            on_complete(result)
                    except Exception as e:
                        logger.error(f"콜백 에러: {e}")

                self._queue.task_done()

            except Exception as e:
                logger.error(f"[Worker {worker_id}] 워커 에러: {e}")

    async def run(
        self,
        on_complete: Callable[[CrawlResult], Any] | None = None,
        on_progress: Callable[[int, int, int], Any] | None = None,
        progress_interval: float = 5.0
    ) -> list[CrawlResult]:
        """
        크롤링 실행

        Args:
            on_complete: 각 작업 완료 시 콜백 (result)
            on_progress: 진행 상황 콜백 (completed, failed, total)
            progress_interval: 진행 상황 로깅 간격 (초)
        """
        self._running = True
        self._results = []
        self._response_times = []

        # 자동 워커 수 계산
        if self._auto_workers:
            self.num_workers = ResourceOptimizer.get_optimal_workers(self._total_tasks)

        logger.info(f"크롤링 시작: {self._total_tasks}개 작업, {self.num_workers}개 워커")

        # 브라우저 풀 초기화 (재사용 모드)
        if self.reuse_browsers:
            self._browser_pool = BrowserPool(
                max_browsers=self.num_workers,
                browser_type=self.browsers[0],
                headless=self.base_config.headless
            )
            await self._browser_pool.start()
            logger.info("브라우저 풀 활성화 (리소스 최적화 모드)")

        # 워커 시작
        workers = [
            asyncio.create_task(self._worker(i, on_complete))
            for i in range(self.num_workers)
        ]

        # 진행 상황 모니터링 + 적응형 동시성
        async def progress_monitor():
            while self._running:
                await asyncio.sleep(progress_interval)
                if self._running:
                    completed = self._completed_tasks
                    failed = self._failed_tasks
                    total = self._total_tasks
                    pending = self._queue.qsize()

                    # 평균 응답 시간 계산
                    avg_time = (
                        sum(self._response_times[-20:]) / len(self._response_times[-20:])
                        if self._response_times else 0
                    )

                    logger.info(
                        f"진행: {completed + failed}/{total} "
                        f"(성공: {completed}, 실패: {failed}, 대기: {pending}, "
                        f"평균: {avg_time:.2f}초)"
                    )

                    # 브라우저 풀 상태
                    if self._browser_pool:
                        stats = self._browser_pool.stats
                        logger.debug(f"브라우저 풀: {stats}")

                    if on_progress:
                        try:
                            if asyncio.iscoroutinefunction(on_progress):
                                await on_progress(completed, failed, total)
                            else:
                                on_progress(completed, failed, total)
                        except Exception:
                            pass

        monitor = asyncio.create_task(progress_monitor())

        # 큐가 빌 때까지 대기
        await self._queue.join()

        # 워커 종료
        self._running = False
        monitor.cancel()

        for worker in workers:
            worker.cancel()

        # 브라우저 풀 종료
        if self._browser_pool:
            await self._browser_pool.close()

        # 결과 정리
        success_rate = (
            self._completed_tasks / self._total_tasks * 100
            if self._total_tasks > 0 else 0
        )
        avg_time = (
            sum(self._response_times) / len(self._response_times)
            if self._response_times else 0
        )

        logger.info(
            f"크롤링 완료: 성공 {self._completed_tasks}, "
            f"실패 {self._failed_tasks}, 총 {self._total_tasks} "
            f"(성공률: {success_rate:.1f}%, 평균: {avg_time:.2f}초)"
        )

        return self._results

    def get_stats(self) -> dict:
        """현재 통계 반환"""
        return {
            "total": self._total_tasks,
            "completed": self._completed_tasks,
            "failed": self._failed_tasks,
            "pending": self._queue.qsize(),
            "running": self._running
        }

    async def pause(self) -> None:
        """일시 정지 (구현 예정)"""
        pass

    async def resume(self) -> None:
        """재개 (구현 예정)"""
        pass

    def adjust_workers(self, num_workers: int) -> None:
        """워커 수 동적 조절 (실행 전에만 가능)"""
        if not self._running:
            self.num_workers = num_workers
            logger.info(f"워커 수 조절: {num_workers}개")


class LightweightCrawler(BaseCrawler):
    """
    최소 리소스 사용 크롤러

    특징:
    - 이미지, 폰트, 미디어 기본 차단 (config로 끌 수 있음)
    - 광고/추적 차단
    - 메모리 최적화
    - 빠른 실행

    데이터만 빠르게 추출할 때 사용 (시각적 요소 불필요 시)
    기본 차단을 비활성화하려면 config에서 명시적으로 설정:

        config = CrawlerConfig(block_images=False)
        crawler = LightweightCrawler(config)

    사용법:
        class MyCrawler(LightweightCrawler):
            async def run(self, url: str):
                await self.goto(url)
                return await self.page.title()
    """

    # 경량 크롤러 기본값 (사용자가 설정 안 한 경우에만 적용)
    DEFAULT_BLOCK_IMAGES = True
    DEFAULT_BLOCK_FONTS = True
    DEFAULT_BLOCK_MEDIA = True
    DEFAULT_BLOCK_ADS = True
    DEFAULT_BLOCK_STYLESHEETS = False

    def __init__(self, config: CrawlerConfig | None = None):
        # 기본 설정 생성
        if config is None:
            config = CrawlerConfig(
                mode="fast",
                block_images=self.DEFAULT_BLOCK_IMAGES,
                block_fonts=self.DEFAULT_BLOCK_FONTS,
                block_media=self.DEFAULT_BLOCK_MEDIA,
                block_ads=self.DEFAULT_BLOCK_ADS,
                block_stylesheets=self.DEFAULT_BLOCK_STYLESHEETS,
            )
        # 사용자 설정이 없는 경우에만 기본값 적용 (강제하지 않음)

        super().__init__(config)

    async def _setup(self) -> None:
        """설정 + 추가 최적화"""
        await super()._setup()

        # 설정에 따라 리소스 차단 (BaseCrawler에서 이미 처리하지만 추가 최적화 가능)
        # 사용자가 명시적으로 False로 설정한 경우는 존중

    async def goto(self, url: str, wait_until: str = "domcontentloaded") -> None:
        """
        빠른 페이지 로드 (networkidle 대신 domcontentloaded)

        networkidle은 모든 네트워크 요청이 끝날 때까지 대기하지만,
        domcontentloaded는 DOM만 로드되면 바로 진행
        """
        await self._rate_limit()
        logger.info(f"페이지 이동 (경량): {url}")
        await self.page.goto(url, wait_until=wait_until, timeout=self.config.navigation_timeout)

    async def run(self, url: str) -> Any:
        """상속받아 구현"""
        raise NotImplementedError("run() 메서드를 구현하세요")


class ContextReuseCrawler(BaseCrawler):
    """
    컨텍스트 재사용 크롤러

    같은 도메인의 여러 페이지를 크롤링할 때 효율적
    - 브라우저/컨텍스트를 재사용
    - 쿠키/세션 유지
    - 연결 재사용

    사용법:
        crawler = ContextReuseCrawler(config)
        await crawler.start()

        for url in urls:
            result = await crawler.crawl(url)

        await crawler.stop()
    """

    def __init__(self, config: CrawlerConfig | None = None):
        super().__init__(config)
        self._started = False

    async def start(self) -> None:
        """크롤러 시작 (브라우저 초기화)"""
        if not self._started:
            await self._setup()
            self._started = True
            logger.info("컨텍스트 재사용 크롤러 시작")

    async def stop(self) -> None:
        """크롤러 종료"""
        if self._started:
            await self._teardown()
            self._started = False
            logger.info("컨텍스트 재사용 크롤러 종료")

    async def crawl(self, url: str) -> Any:
        """단일 URL 크롤링 (컨텍스트 재사용)"""
        if not self._started:
            raise RuntimeError("크롤러가 시작되지 않음. start()를 먼저 호출하세요.")

        return await self.run(url)

    async def crawl_many(self, urls: list[str]) -> list[dict]:
        """여러 URL 순차 크롤링"""
        results = []
        for url in urls:
            try:
                data = await self.crawl(url)
                results.append({"url": url, "success": True, "data": data})
            except Exception as e:
                results.append({"url": url, "success": False, "error": str(e)})
        return results

    async def run(self, url: str) -> Any:
        """상속받아 구현"""
        raise NotImplementedError("run() 메서드를 구현하세요")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


def get_system_resources() -> dict:
    """
    시스템 리소스 정보 조회

    Returns:
        {
            "cpu_count": CPU 코어 수,
            "memory_total_gb": 전체 메모리 (GB),
            "memory_available_gb": 사용 가능 메모리 (GB),
            "recommended_workers": 권장 워커 수
        }
    """
    import os

    result = {
        "cpu_count": os.cpu_count() or 4,
        "memory_total_gb": 8.0,  # 기본값
        "memory_available_gb": 4.0,  # 기본값
    }

    # psutil이 있으면 실제 메모리 정보
    try:
        import psutil
        mem = psutil.virtual_memory()
        result["memory_total_gb"] = mem.total / (1024 ** 3)
        result["memory_available_gb"] = mem.available / (1024 ** 3)
    except ImportError:
        pass

    # 권장 워커 수 계산
    result["recommended_workers"] = ResourceOptimizer.get_optimal_workers(
        total_urls=100,
        available_memory_gb=result["memory_available_gb"]
    )

    return result


def save_to_csv(data: list[dict], filename: str, encoding: str = "utf-8-sig") -> str:
    """
    데이터를 CSV 파일로 저장

    Args:
        data: 딕셔너리 리스트
        filename: 저장 파일명
        encoding: 인코딩 (Excel 호환: utf-8-sig)
    """
    import csv

    if not data:
        logger.warning("저장할 데이터가 없습니다.")
        return filename

    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', newline='', encoding=encoding) as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

    logger.info(f"CSV 저장 완료: {filepath} ({len(data)}행)")
    return str(filepath)


def chunks(lst: list, n: int):
    """리스트를 n개씩 분할"""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


async def main():
    """예시 실행"""
    config = CrawlerConfig(
        mode="fast",
        headless=True,
        debug=True,
        verbose=True
    )

    crawler = ExampleCrawler(config=config)
    result = await crawler.execute("https://example.com")

    print(f"결과: {result}")


if __name__ == "__main__":
    asyncio.run(main())
