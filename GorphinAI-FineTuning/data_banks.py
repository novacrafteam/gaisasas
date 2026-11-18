"""Template-based knowledge/behaviour banks for the dataset generator.

Every entry is hand-written in uk / ru / en. Facts were kept to well-established basics.
All data produced from here is SYNTHETIC (it is labelled source="template" in the output).
"""

# ------------------------------------------------------------------ chat
CHAT = [
    {"uk": ("Привіт! Як справи?", "Привіт! У мене все добре, дякую. А в тебе як? Чим можу допомогти сьогодні?"),
     "ru": ("Привет! Как дела?", "Привет! У меня всё хорошо, спасибо. А у тебя как? Чем могу помочь сегодня?"),
     "en": ("Hi! How are you?", "Hi! I'm doing well, thanks. How about you? What can I help you with today?")},
    {"uk": ("Хто ти?", "Я GorphinAI — AI-асистент. Допомагаю з програмуванням, математикою, навчанням і поясненням складних речей. Я можу помилятися, тому важливі факти варто перевіряти."),
     "ru": ("Кто ты?", "Я GorphinAI — AI-ассистент. Помогаю с программированием, математикой, учёбой и объяснением сложных вещей. Я могу ошибаться, поэтому важные факты стоит перепроверять."),
     "en": ("Who are you?", "I'm GorphinAI, an AI assistant. I help with programming, math, studying and explaining complex things. I can make mistakes, so it's worth double-checking important facts.")},
    {"uk": ("Мені сьогодні сумно.", "Шкода це чути. Якщо хочеш, розкажи, що сталося — я вислухаю. А якщо хочеться відволіктися, можемо просто поговорити про щось приємне. Пам'ятай: якщо цей стан триває довго, варто поговорити з близькою людиною або фахівцем."),
     "ru": ("Мне сегодня грустно.", "Жаль это слышать. Если хочешь, расскажи, что случилось — я выслушаю. А если хочется отвлечься, можем просто поговорить о чём-нибудь приятном. И помни: если такое состояние длится долго, стоит поговорить с близким человеком или специалистом."),
     "en": ("I feel sad today.", "I'm sorry to hear that. If you'd like, tell me what happened — I'm happy to listen. If you'd rather get your mind off it, we can just talk about something pleasant. And if this feeling lasts for a long time, it's worth talking to someone close to you or a professional.")},
    {"uk": ("Дякую за допомогу!", "Будь ласка! Радий, що вдалося допомогти. Якщо з'являться ще питання — пиши."),
     "ru": ("Спасибо за помощь!", "Пожалуйста! Рад, что получилось помочь. Если появятся ещё вопросы — пиши."),
     "en": ("Thanks for the help!", "You're welcome! Glad I could help. If more questions come up, just ask.")},
    {"uk": ("Яка сьогодні погода?", "Я не маю доступу до актуальних даних про погоду, тому не можу сказати точно. Подивись прогноз у погодному застосунку або на сайті метеослужби."),
     "ru": ("Какая сегодня погода?", "У меня нет доступа к актуальным данным о погоде, поэтому точно сказать не могу. Посмотри прогноз в погодном приложении или на сайте метеослужбы."),
     "en": ("What's the weather like today?", "I don't have access to live weather data, so I can't tell you accurately. Check a weather app or your local meteorological service.")},
    {"uk": ("Порадь, як краще організувати свій день.", "Ось простий підхід: 1) вибери 1–3 головні задачі на день; 2) став їх на час, коли ти найпродуктивніший; 3) роби короткі перерви кожні 45–90 хвилин; 4) ввечері коротко підведи підсумок. Якщо скажеш, чим ти займаєшся, підлаштую план під тебе."),
     "ru": ("Посоветуй, как лучше организовать свой день.", "Вот простой подход: 1) выбери 1–3 главные задачи на день; 2) поставь их на время, когда ты наиболее продуктивен; 3) делай короткие перерывы каждые 45–90 минут; 4) вечером коротко подведи итог. Если расскажешь, чем занимаешься, подстрою план под тебя."),
     "en": ("Give me advice on organizing my day.", "Here's a simple approach: 1) pick 1–3 main tasks for the day; 2) schedule them for when you're most productive; 3) take short breaks every 45–90 minutes; 4) briefly review the day in the evening. If you tell me what you do, I'll tailor the plan to you.")},
]

# ------------------------------------------------------------------ python / js code tasks
# request phrasing variants per language, then code (language-neutral) + explanation per language
PY_REQ = {
    "uk": ["Напиши на Python {t}.", "Як на Python {t}?", "Допоможи написати код на Python: {t}."],
    "ru": ["Напиши на Python {t}.", "Как на Python {t}?", "Помоги написать код на Python: {t}."],
    "en": ["Write Python code to {t}.", "How do I {t} in Python?", "Help me with Python: {t}."],
}
JS_REQ = {
    "uk": ["Напиши на JavaScript {t}.", "Як на JavaScript {t}?", "Допоможи з кодом на JavaScript: {t}."],
    "ru": ["Напиши на JavaScript {t}.", "Как на JavaScript {t}?", "Помоги с кодом на JavaScript: {t}."],
    "en": ["Write JavaScript code to {t}.", "How do I {t} in JavaScript?", "Help me with JavaScript: {t}."],
}

PY_TASKS = [
    {"t": {"uk": "функцію, що перевертає рядок", "ru": "функцию, которая переворачивает строку", "en": "reverse a string"},
     "code": "def reverse_string(s: str) -> str:\n    return s[::-1]\n\nprint(reverse_string(\"hello\"))  # olleh",
     "x": {"uk": "Зріз `[::-1]` бере елементи з кроком -1, тобто у зворотному порядку.",
           "ru": "Срез `[::-1]` берёт элементы с шагом -1, то есть в обратном порядке.",
           "en": "The slice `[::-1]` walks the string with step -1, i.e. in reverse order."}},
    {"t": {"uk": "функцію, що перевіряє, чи є число простим", "ru": "функцию, которая проверяет, является ли число простым", "en": "check whether a number is prime"},
     "code": "def is_prime(n: int) -> bool:\n    if n < 2:\n        return False\n    i = 2\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i += 1\n    return True\n\nprint(is_prime(17))  # True",
     "x": {"uk": "Достатньо перевіряти дільники до √n, бо більший дільник завжди має пару серед менших.",
           "ru": "Достаточно проверять делители до √n, потому что у большего делителя всегда есть пара среди меньших.",
           "en": "It's enough to test divisors up to √n, because every larger divisor has a partner among the smaller ones."}},
    {"t": {"uk": "функцію для обчислення чисел Фібоначчі", "ru": "функцию для вычисления чисел Фибоначчи", "en": "compute Fibonacci numbers"},
     "code": "def fib(n: int) -> int:\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n\nprint([fib(i) for i in range(10)])  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]",
     "x": {"uk": "Ітеративна версія працює за O(n) і не має обмеження глибини рекурсії.",
           "ru": "Итеративная версия работает за O(n) и не упирается в лимит глубины рекурсии.",
           "en": "The iterative version runs in O(n) and avoids recursion depth limits."}},
    {"t": {"uk": "код, що рахує частоту слів у тексті", "ru": "код, который считает частоту слов в тексте", "en": "count word frequencies in a text"},
     "code": "from collections import Counter\n\ntext = \"the cat and the dog and the bird\"\ncounts = Counter(text.lower().split())\nprint(counts.most_common(2))  # [('the', 3), ('and', 2)]",
     "x": {"uk": "`Counter` рахує кількість повторень, а `most_common(n)` повертає n найчастіших елементів.",
           "ru": "`Counter` считает количество повторений, а `most_common(n)` возвращает n самых частых элементов.",
           "en": "`Counter` counts occurrences and `most_common(n)` returns the n most frequent items."}},
    {"t": {"uk": "код, що читає JSON-файл", "ru": "код, который читает JSON-файл", "en": "read a JSON file"},
     "code": "import json\n\nwith open(\"data.json\", encoding=\"utf-8\") as f:\n    data = json.load(f)\n\nprint(type(data))",
     "x": {"uk": "Конструкція `with` сама закриє файл, а `encoding=\"utf-8\"` захищає від проблем із кирилицею.",
           "ru": "Конструкция `with` сама закроет файл, а `encoding=\"utf-8\"` защищает от проблем с кириллицей.",
           "en": "`with` closes the file automatically and `encoding=\"utf-8\"` avoids problems with non-ASCII text."}},
    {"t": {"uk": "код, що прибирає дублікати зі списку зі збереженням порядку", "ru": "код, который убирает дубликаты из списка с сохранением порядка", "en": "remove duplicates from a list while keeping order"},
     "code": "items = [3, 1, 3, 2, 1]\nunique = list(dict.fromkeys(items))\nprint(unique)  # [3, 1, 2]",
     "x": {"uk": "Словники в Python зберігають порядок вставки, тому `dict.fromkeys` дає унікальні елементи в початковому порядку.",
           "ru": "Словари в Python сохраняют порядок вставки, поэтому `dict.fromkeys` даёт уникальные элементы в исходном порядке.",
           "en": "Python dicts keep insertion order, so `dict.fromkeys` yields unique items in their original order."}},
    {"t": {"uk": "функцію, що перевіряє паліндром", "ru": "функцию, которая проверяет палиндром", "en": "check if a string is a palindrome"},
     "code": "def is_palindrome(s: str) -> bool:\n    cleaned = [c.lower() for c in s if c.isalnum()]\n    return cleaned == cleaned[::-1]\n\nprint(is_palindrome(\"A man, a plan, a canal: Panama\"))  # True",
     "x": {"uk": "Спочатку лишаємо тільки літери й цифри в нижньому регістрі, потім порівнюємо з її реверсом.",
           "ru": "Сначала оставляем только буквы и цифры в нижнем регистре, затем сравниваем с её реверсом.",
           "en": "We first keep only lowercase letters and digits, then compare the sequence with its reverse."}},
    {"t": {"uk": "бінарний пошук у відсортованому списку", "ru": "бинарный поиск в отсортированном списке", "en": "do binary search in a sorted list"},
     "code": "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        if arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n\nprint(binary_search([1, 3, 5, 7, 9], 7))  # 3",
     "x": {"uk": "Щоразу відкидаємо половину діапазону, тому складність O(log n). Список має бути відсортованим.",
           "ru": "Каждый раз отбрасываем половину диапазона, поэтому сложность O(log n). Список должен быть отсортирован.",
           "en": "Each step discards half of the range, so the complexity is O(log n). The list must be sorted."}},
]

JS_TASKS = [
    {"t": {"uk": "функцію, що перевертає рядок", "ru": "функцию, которая переворачивает строку", "en": "reverse a string"},
     "code": "const reverse = (s) => [...s].reverse().join(\"\");\n\nconsole.log(reverse(\"hello\")); // olleh",
     "x": {"uk": "Розгортання `[...s]` розбиває рядок на символи, `reverse()` розвертає масив, `join` склеює назад.",
           "ru": "Распаковка `[...s]` разбивает строку на символы, `reverse()` разворачивает массив, `join` склеивает обратно.",
           "en": "Spreading `[...s]` splits the string into characters, `reverse()` flips the array and `join` glues it back."}},
    {"t": {"uk": "суму всіх елементів масиву", "ru": "сумму всех элементов массива", "en": "sum all elements of an array"},
     "code": "const sum = (arr) => arr.reduce((acc, x) => acc + x, 0);\n\nconsole.log(sum([1, 2, 3, 4])); // 10",
     "x": {"uk": "`reduce` накопичує результат; початкове значення `0` робить код безпечним і для порожнього масиву.",
           "ru": "`reduce` накапливает результат; начальное значение `0` делает код безопасным и для пустого массива.",
           "en": "`reduce` accumulates a result; the initial value `0` keeps it safe for an empty array."}},
    {"t": {"uk": "завантаження JSON з сервера через fetch", "ru": "загрузку JSON с сервера через fetch", "en": "fetch JSON from a server"},
     "code": "async function getJson(url) {\n  const res = await fetch(url);\n  if (!res.ok) {\n    throw new Error(`HTTP ${res.status}`);\n  }\n  return res.json();\n}\n\ngetJson(\"https://example.com/api\").then(console.log).catch(console.error);",
     "x": {"uk": "`fetch` не кидає помилку на статуси 4xx/5xx, тому перевіряємо `res.ok` самі.",
           "ru": "`fetch` не бросает ошибку на статусах 4xx/5xx, поэтому проверяем `res.ok` сами.",
           "en": "`fetch` does not reject on 4xx/5xx statuses, so we check `res.ok` ourselves."}},
    {"t": {"uk": "фільтрацію парних чисел у масиві", "ru": "фильтрацию чётных чисел в массиве", "en": "filter even numbers from an array"},
     "code": "const evens = [1, 2, 3, 4, 5, 6].filter((n) => n % 2 === 0);\nconsole.log(evens); // [2, 4, 6]",
     "x": {"uk": "`filter` залишає лише елементи, для яких callback повертає true.",
           "ru": "`filter` оставляет только элементы, для которых callback возвращает true.",
           "en": "`filter` keeps only the elements for which the callback returns true."}},
    {"t": {"uk": "debounce-функцію", "ru": "debounce-функцию", "en": "write a debounce function"},
     "code": "function debounce(fn, delay) {\n  let timer;\n  return (...args) => {\n    clearTimeout(timer);\n    timer = setTimeout(() => fn(...args), delay);\n  };\n}\n\nconst onType = debounce((q) => console.log(\"search\", q), 300);",
     "x": {"uk": "Кожен новий виклик скидає таймер, тож `fn` виконається лише після паузи в `delay` мс.",
           "ru": "Каждый новый вызов сбрасывает таймер, поэтому `fn` выполнится только после паузы в `delay` мс.",
           "en": "Every new call resets the timer, so `fn` runs only after a pause of `delay` ms."}},
    {"t": {"uk": "групування об'єктів за полем", "ru": "группировку объектов по полю", "en": "group objects by a field"},
     "code": "const groupBy = (arr, key) =>\n  arr.reduce((acc, item) => {\n    (acc[item[key]] ||= []).push(item);\n    return acc;\n  }, {});\n\nconsole.log(groupBy([{t: \"a\"}, {t: \"b\"}, {t: \"a\"}], \"t\"));",
     "x": {"uk": "Для кожного елемента беремо масив за ключем (створюємо, якщо його ще немає) і додаємо елемент.",
           "ru": "Для каждого элемента берём массив по ключу (создаём, если его ещё нет) и добавляем элемент.",
           "en": "For each item we take the array under its key (creating it if missing) and push the item."}},
]

# ------------------------------------------------------------------ generic knowledge categories
PROGRAMMING = [
    {"uk": ("У чому різниця між стеком і чергою?", "Стек працює за принципом LIFO (останнім прийшов — першим пішов), черга — за FIFO (першим прийшов — першим пішов). Стек — як стопка тарілок, черга — як черга в магазині."),
     "ru": ("В чём разница между стеком и очередью?", "Стек работает по принципу LIFO (последним пришёл — первым ушёл), очередь — по FIFO (первым пришёл — первым ушёл). Стек — как стопка тарелок, очередь — как очередь в магазине."),
     "en": ("What's the difference between a stack and a queue?", "A stack is LIFO (last in, first out) and a queue is FIFO (first in, first out). A stack is like a pile of plates; a queue is like a line at a shop.")},
    {"uk": ("Що таке Git і навіщо він потрібен?", "Git — це система контролю версій. Вона зберігає історію змін коду, дозволяє працювати в гілках, повертатися до попередніх версій і зручно співпрацювати з іншими людьми."),
     "ru": ("Что такое Git и зачем он нужен?", "Git — это система контроля версий. Она хранит историю изменений кода, позволяет работать в ветках, возвращаться к прежним версиям и удобно сотрудничать с другими людьми."),
     "en": ("What is Git and why do I need it?", "Git is a version control system. It stores the history of your code changes, lets you work in branches, go back to earlier versions and collaborate easily with others.")},
    {"uk": ("Що таке рекурсія?", "Рекурсія — це коли функція викликає сама себе для розв'язання меншої версії тієї ж задачі. Обов'язково потрібна умова виходу (базовий випадок), інакше буде нескінченна рекурсія."),
     "ru": ("Что такое рекурсия?", "Рекурсия — это когда функция вызывает саму себя для решения меньшей версии той же задачи. Обязательно нужно условие выхода (базовый случай), иначе получится бесконечная рекурсия."),
     "en": ("What is recursion?", "Recursion is when a function calls itself to solve a smaller version of the same problem. It must have a base case, otherwise it recurses forever.")},
    {"uk": ("Що означає O(n) у програмуванні?", "Це оцінка складності алгоритму: час роботи зростає приблизно лінійно від розміру вхідних даних n. Якщо даних удвічі більше — часу потрібно приблизно удвічі більше."),
     "ru": ("Что значит O(n) в программировании?", "Это оценка сложности алгоритма: время работы растёт примерно линейно от размера входных данных n. Если данных вдвое больше — времени нужно примерно вдвое больше."),
     "en": ("What does O(n) mean in programming?", "It's an estimate of algorithm complexity: running time grows roughly linearly with the input size n. Twice the data takes roughly twice the time.")},
    {"uk": ("Як шукати помилки в коді, якщо не розумію, що не так?", "Спробуй такий порядок: 1) уважно прочитай текст помилки та номер рядка; 2) відтвори проблему на мінімальному прикладі; 3) виведи проміжні значення (print або дебагер); 4) перевір припущення по одному. Якщо покажеш код і помилку, розберемо разом."),
     "ru": ("Как искать ошибки в коде, если не понимаю, что не так?", "Попробуй такой порядок: 1) внимательно прочитай текст ошибки и номер строки; 2) воспроизведи проблему на минимальном примере; 3) выведи промежуточные значения (print или отладчик); 4) проверяй предположения по одному. Если покажешь код и ошибку, разберём вместе."),
     "en": ("How do I find bugs when I don't understand what's wrong?", "Try this order: 1) read the error message and line number carefully; 2) reproduce the problem in a minimal example; 3) print intermediate values (or use a debugger); 4) test your assumptions one at a time. If you show me the code and the error, we can go through it together.")},
]

SCIENCE = [
    {"uk": ("Чому небо блакитне?", "Через розсіювання Релея: молекули атмосфери розсіюють короткі хвилі (синій колір) сильніше, ніж довгі. Тому розсіяне синє світло приходить до нас з усіх боків неба."),
     "ru": ("Почему небо голубое?", "Из-за рэлеевского рассеяния: молекулы атмосферы рассеивают короткие волны (синий цвет) сильнее, чем длинные. Поэтому рассеянный синий свет приходит к нам со всех сторон неба."),
     "en": ("Why is the sky blue?", "Because of Rayleigh scattering: air molecules scatter short wavelengths (blue) much more strongly than long ones, so scattered blue light reaches us from all directions of the sky.")},
    {"uk": ("Що таке фотосинтез?", "Це процес, у якому рослини, водорості та деякі бактерії використовують світлову енергію, щоб із вуглекислого газу та води створювати органічні речовини (глюкозу). Побічно виділяється кисень."),
     "ru": ("Что такое фотосинтез?", "Это процесс, в котором растения, водоросли и некоторые бактерии используют энергию света, чтобы из углекислого газа и воды создавать органические вещества (глюкозу). Побочно выделяется кислород."),
     "en": ("What is photosynthesis?", "It's the process by which plants, algae and some bacteria use light energy to turn carbon dioxide and water into organic matter (glucose). Oxygen is released as a by-product.")},
    {"uk": ("Чому Місяць має фази?", "Місяць сам не світить, а відбиває світло Сонця. Коли він обертається навколо Землі, ми бачимо різну частину його освітленої половини — звідси фази від новомісяця до повні."),
     "ru": ("Почему у Луны бывают фазы?", "Луна сама не светит, а отражает свет Солнца. Когда она обращается вокруг Земли, мы видим разную часть её освещённой половины — отсюда фазы от новолуния до полнолуния."),
     "en": ("Why does the Moon have phases?", "The Moon doesn't produce light; it reflects sunlight. As it orbits the Earth we see different portions of its lit half, which gives phases from new moon to full moon.")},
    {"uk": ("З чого складається вода?", "Молекула води H₂O складається з двох атомів водню та одного атома кисню, з'єднаних ковалентними зв'язками."),
     "ru": ("Из чего состоит вода?", "Молекула воды H₂O состоит из двух атомов водорода и одного атома кислорода, соединённых ковалентными связями."),
     "en": ("What is water made of?", "A water molecule, H₂O, consists of two hydrogen atoms and one oxygen atom joined by covalent bonds.")},
    {"uk": ("Яка швидкість світла?", "У вакуумі приблизно 299 792 458 м/с, тобто близько 300 000 км/с."),
     "ru": ("Какова скорость света?", "В вакууме примерно 299 792 458 м/с, то есть около 300 000 км/с."),
     "en": ("What is the speed of light?", "In a vacuum it is exactly 299,792,458 m/s, roughly 300,000 km/s.")},
]

TECH = [
    {"uk": ("Що таке API простими словами?", "API — це «меню» програми: набір чітких запитів, через які одна програма може попросити іншу щось зробити або віддати дані, не знаючи, як та влаштована всередині."),
     "ru": ("Что такое API простыми словами?", "API — это «меню» программы: набор чётких запросов, через которые одна программа может попросить другую что-то сделать или отдать данные, не зная, как та устроена внутри."),
     "en": ("What is an API in simple terms?", "An API is a program's \"menu\": a set of well-defined requests through which one program can ask another to do something or return data without knowing how it works inside.")},
    {"uk": ("Яка різниця між RAM і SSD?", "RAM — швидка тимчасова пам'ять, у якій програми тримають дані під час роботи; після вимкнення вона очищується. SSD — повільніший, але постійний накопичувач, де файли зберігаються й без живлення."),
     "ru": ("В чём разница между RAM и SSD?", "RAM — быстрая временная память, в которой программы держат данные во время работы; после выключения она очищается. SSD — более медленный, но постоянный накопитель, где файлы хранятся и без питания."),
     "en": ("What's the difference between RAM and an SSD?", "RAM is fast, temporary memory where programs keep data while running; it is cleared at power-off. An SSD is slower but persistent storage that keeps files without power.")},
    {"uk": ("Що таке VPN?", "VPN створює зашифрований тунель між твоїм пристроєм і VPN-сервером. Сайти бачать IP сервера, а провайдер — лише зашифрований трафік. Це не робить тебе повністю анонімним, а лише змінює, хто що бачить."),
     "ru": ("Что такое VPN?", "VPN создаёт зашифрованный туннель между твоим устройством и VPN-сервером. Сайты видят IP сервера, а провайдер — только зашифрованный трафик. Это не делает тебя полностью анонимным, а лишь меняет, кто что видит."),
     "en": ("What is a VPN?", "A VPN creates an encrypted tunnel between your device and a VPN server. Websites see the server's IP and your ISP sees only encrypted traffic. It doesn't make you fully anonymous; it just changes who can see what.")},
    {"uk": ("Що таке нейронна мережа?", "Це модель машинного навчання з шарів простих обчислювальних вузлів (нейронів), які перетворюють вхідні дані. Під час навчання підбираються ваги зв'язків так, щоб помилка передбачень зменшувалась."),
     "ru": ("Что такое нейронная сеть?", "Это модель машинного обучения из слоёв простых вычислительных узлов (нейронов), которые преобразуют входные данные. При обучении подбираются веса связей так, чтобы ошибка предсказаний уменьшалась."),
     "en": ("What is a neural network?", "It's a machine-learning model made of layers of simple computing units (neurons) that transform the input. During training the connection weights are adjusted so prediction error decreases.")},
]

EDUCATION = [
    {"uk": ("Як ефективно вчити іноземні слова?", "Працює поєднання трьох речей: інтервальні повторення (наприклад, картки), слова в контексті речень і активне використання (говорити чи писати). Краще 15 хвилин щодня, ніж 2 години раз на тиждень."),
     "ru": ("Как эффективно учить иностранные слова?", "Работает сочетание трёх вещей: интервальные повторения (например, карточки), слова в контексте предложений и активное использование (говорить или писать). Лучше 15 минут каждый день, чем 2 часа раз в неделю."),
     "en": ("How can I learn foreign words effectively?", "Three things work well together: spaced repetition (e.g. flashcards), words in sentence context, and active use (speaking or writing). 15 minutes daily beats 2 hours once a week.")},
    {"uk": ("Як підготуватися до іспиту за тиждень?", "Розбий матеріал на теми за днями, залиши останній день на повторення. Замість перечитування перевіряй себе питаннями й розв'язуй приклади завдань. Спи достатньо — це суттєво впливає на запам'ятовування."),
     "ru": ("Как подготовиться к экзамену за неделю?", "Разбей материал на темы по дням, оставь последний день на повторение. Вместо перечитывания проверяй себя вопросами и решай примеры заданий. Высыпайся — это существенно влияет на запоминание."),
     "en": ("How do I prepare for an exam in a week?", "Split the material into topics by day and leave the last day for review. Instead of rereading, quiz yourself and solve practice problems. Get enough sleep — it strongly affects memory.")},
    {"uk": ("З чого почати вивчення програмування?", "Обери одну мову (для старту добре підходить Python), пройди базовий курс і одразу пиши маленькі проєкти: калькулятор, список справ, бот. Практика важливіша за кількість прочитаних уроків."),
     "ru": ("С чего начать изучение программирования?", "Выбери один язык (для старта хорошо подходит Python), пройди базовый курс и сразу пиши маленькие проекты: калькулятор, список дел, бот. Практика важнее количества прочитанных уроков."),
     "en": ("Where should I start learning programming?", "Pick one language (Python is a good first choice), take a basic course and immediately build small projects: a calculator, a to-do list, a bot. Practice matters more than the number of lessons read.")},
    {"uk": ("Як перестати відкладати навчання на потім?", "Зменш поріг входу: пообіцяй собі лише 5 хвилин. Прибери відволікання (телефон в іншу кімнату) і прив'яжи заняття до фіксованого часу. Починати зазвичай найважче — далі йде легше."),
     "ru": ("Как перестать откладывать учёбу на потом?", "Снизь порог входа: пообещай себе всего 5 минут. Убери отвлекающие факторы (телефон в другую комнату) и привяжи занятие к фиксированному времени. Начинать обычно тяжелее всего — дальше идёт легче."),
     "en": ("How do I stop procrastinating on studying?", "Lower the barrier: promise yourself just 5 minutes. Remove distractions (phone in another room) and tie studying to a fixed time. Starting is usually the hardest part; it gets easier afterwards.")},
]

EXPLAIN = [
    {"uk": ("Поясни, що таке інфляція, як для п'ятикласника.", "Уяви, що минулого року морозиво коштувало 20 гривень, а цього — 22. За ті самі гроші ти тепер можеш купити менше. Коли ціни загалом зростають, а гроші «слабшають» — це і є інфляція."),
     "ru": ("Объясни, что такое инфляция, как для пятиклассника.", "Представь, что в прошлом году мороженое стоило 20 рублей, а в этом — 22. На те же деньги ты теперь можешь купить меньше. Когда цены в целом растут, а деньги «слабеют» — это и есть инфляция."),
     "en": ("Explain inflation like I'm ten.", "Imagine ice cream cost $2.00 last year and $2.20 now. The same money buys you less. When prices in general keep going up and money \"gets weaker\", that's inflation.")},
    {"uk": ("Поясни простими словами, як працює інтернет.", "Інтернет — це величезна мережа з'єднаних комп'ютерів. Коли ти відкриваєш сайт, твій пристрій надсилає запит, він розбивається на маленькі пакети, які різними маршрутами через маршрутизатори доходять до сервера. Сервер відповідає так само, а пристрій збирає пакети назад у сторінку."),
     "ru": ("Объясни простыми словами, как работает интернет.", "Интернет — это огромная сеть соединённых компьютеров. Когда ты открываешь сайт, твоё устройство отправляет запрос, он разбивается на маленькие пакеты, которые разными маршрутами через маршрутизаторы доходят до сервера. Сервер отвечает так же, а устройство собирает пакеты обратно в страницу."),
     "en": ("Explain in simple words how the internet works.", "The internet is a huge network of connected computers. When you open a website, your device sends a request that is split into small packets, which travel via routers, possibly on different routes, to the server. The server replies the same way and your device reassembles the packets into the page.")},
    {"uk": ("Що таке блокчейн, без складних термінів?", "Це спільний журнал записів, копія якого є в багатьох учасників. Нові записи додаються блоками й прив'язуються до попередніх, тому змінити старий запис непомітно майже неможливо — довелося б переписати його в більшості копій."),
     "ru": ("Что такое блокчейн, без сложных терминов?", "Это общий журнал записей, копия которого есть у многих участников. Новые записи добавляются блоками и привязываются к предыдущим, поэтому изменить старую запись незаметно почти невозможно — пришлось бы переписать её в большинстве копий."),
     "en": ("What is a blockchain, without jargon?", "It's a shared ledger of records that many participants hold copies of. New records are added in blocks linked to the previous ones, so changing an old record unnoticed is almost impossible — you'd have to rewrite it in most copies.")},
]

ANALYSIS = [
    {"uk": ("Проаналізуй: продажі в січні — 120, у лютому — 150, у березні — 135. Що можна сказати?", "Від січня до лютого продажі зросли на 25% (120 → 150), у березні знизилися на 10% (150 → 135). Загалом за квартал є зростання відносно січня (+12,5%), але динаміка нерівна. Трьох точок замало для висновку про тренд — варто подивитися більший період і причини коливань (сезонність, акції)."),
     "ru": ("Проанализируй: продажи в январе — 120, в феврале — 150, в марте — 135. Что можно сказать?", "От января к февралю продажи выросли на 25% (120 → 150), в марте снизились на 10% (150 → 135). В целом за квартал есть рост относительно января (+12,5%), но динамика неровная. Трёх точек мало для вывода о тренде — стоит посмотреть больший период и причины колебаний (сезонность, акции)."),
     "en": ("Analyze: sales were 120 in January, 150 in February and 135 in March. What can you say?", "From January to February sales grew by 25% (120 → 150) and in March they fell by 10% (150 → 135). Over the quarter they are up 12.5% relative to January, but the dynamics are uneven. Three data points are too few to claim a trend — look at a longer period and the causes of fluctuations (seasonality, promotions).")},
    {"uk": ("У групи А середній бал 4,2, у групі Б — 4,5. Чи означає це, що група Б навчається краще?", "Не обов'язково. Потрібно знати розмір груп, розкид оцінок і чи однакові умови оцінювання. Різниця в 0,3 може бути випадковою, якщо груп малі. Для обґрунтованого висновку варто порівняти розподіли та, за можливості, застосувати статистичний тест."),
     "ru": ("В группе А средний балл 4,2, в группе Б — 4,5. Значит ли это, что группа Б учится лучше?", "Не обязательно. Нужно знать размеры групп, разброс оценок и одинаковы ли условия оценивания. Разница в 0,3 может быть случайной, если группы малы. Для обоснованного вывода стоит сравнить распределения и, по возможности, применить статистический тест."),
     "en": ("Group A has an average grade of 4.2, group B has 4.5. Does that mean group B performs better?", "Not necessarily. You need the group sizes, the spread of grades and whether the grading conditions are the same. A difference of 0.3 can be random if the groups are small. For a sound conclusion, compare the distributions and, if possible, run a statistical test.")},
]

PROBLEM = [
    {"uk": ("Мій комп'ютер дуже повільно працює. Що перевірити?", "Перевір по черзі: 1) Диспетчер завдань — що навантажує процесор, пам'ять чи диск; 2) автозавантаження — вимкни зайве; 3) вільне місце на диску (бажано >15%); 4) оновлення та віруси; 5) чи не перегрівається пристрій. Якщо скажеш модель і що саме гальмує, звузимо пошук."),
     "ru": ("Мой компьютер очень медленно работает. Что проверить?", "Проверь по очереди: 1) Диспетчер задач — что нагружает процессор, память или диск; 2) автозагрузка — отключи лишнее; 3) свободное место на диске (желательно >15%); 4) обновления и вирусы; 5) не перегревается ли устройство. Если скажешь модель и что именно тормозит, сузим поиск."),
     "en": ("My computer is running very slowly. What should I check?", "Check in this order: 1) Task Manager — what loads the CPU, memory or disk; 2) startup programs — disable unneeded ones; 3) free disk space (ideally >15%); 4) updates and malware; 5) whether the device overheats. If you tell me the model and what exactly is slow, we can narrow it down.")},
    {"uk": ("Як вирішити, чи варто змінювати роботу?", "Випиши, що тебе не влаштовує зараз і чого ти хочеш (гроші, розвиток, графік, команда). Оціни, чи це можна змінити на поточному місці. Подивись на реальні пропозиції ринку й запас фінансів на пошук. Якщо хочеш, розкажи деталі — допоможу структурувати."),
     "ru": ("Как решить, стоит ли менять работу?", "Выпиши, что тебя не устраивает сейчас и чего ты хочешь (деньги, развитие, график, команда). Оцени, можно ли это изменить на текущем месте. Посмотри на реальные предложения рынка и запас финансов на поиск. Если хочешь, расскажи детали — помогу структурировать."),
     "en": ("How do I decide whether to change jobs?", "Write down what bothers you now and what you want (pay, growth, schedule, team). Check whether it can be changed in your current role. Look at real market offers and your financial runway for the search. If you like, share details and I'll help you structure it.")},
]

TEXT_WORK = [
    {"uk": ("Виправ помилки: «Я йшов до магазину і купив хліб, молоко і яйця.»", "Речення вже правильне: «Я йшов до магазину і купив хліб, молоко і яйця.» Помилок немає. За бажання можна сказати коротше: «Я пішов до магазину й купив хліб, молоко та яйця.»"),
     "ru": ("Исправь ошибки: «Я шёл в магазин и купил хлеб, молоко и яйца.»", "Предложение уже правильное: «Я шёл в магазин и купил хлеб, молоко и яйца.» Ошибок нет. При желании можно сделать короче: «Я пошёл в магазин и купил хлеб, молоко и яйца.»"),
     "en": ("Fix the mistakes: \"I goes to the shop and buyed some bread.\"", "Corrected: \"I went to the shop and bought some bread.\" The verbs needed the past tense: \"go\" → \"went\" and \"buy\" → \"bought\" (both irregular).")},
    {"uk": ("Стисни в одне речення: «Сьогодні ми обговорили бюджет, розподілили задачі між командою та домовилися про наступну зустріч у п'ятницю.»", "На зустрічі ми погодили бюджет, розподілили задачі та призначили наступну зустріч на п'ятницю."),
     "ru": ("Сожми в одно предложение: «Сегодня мы обсудили бюджет, распределили задачи между командой и договорились о следующей встрече в пятницу.»", "На встрече мы согласовали бюджет, распределили задачи и назначили следующую встречу на пятницу."),
     "en": ("Condense into one sentence: \"Today we discussed the budget, assigned tasks among the team and agreed on the next meeting on Friday.\"", "At the meeting we settled the budget, assigned tasks and scheduled the next meeting for Friday.")},
    {"uk": ("Зроби цей текст більш ввічливим: «Дай мені звіт до завтра.»", "Варіант: «Будь ласка, чи не міг би ти надіслати мені звіт до завтра? Дякую!»"),
     "ru": ("Сделай этот текст более вежливым: «Дай мне отчёт до завтра.»", "Вариант: «Пожалуйста, не мог бы ты прислать мне отчёт до завтра? Спасибо!»"),
     "en": ("Make this more polite: \"Give me the report by tomorrow.\"", "Option: \"Could you please send me the report by tomorrow? Thank you!\"")},
]

HONESTY = [  # teaches not to fabricate
    {"uk": ("Скільки людей буде жити на Землі у 2150 році?", "Точно цього ніхто не знає. Демографічні прогнози на такий строк дуже невизначені й залежать від народжуваності, медицини та подій, яких ми не можемо передбачити. Тому я не назву конкретну цифру як факт — можу розповісти, як роблять такі прогнози."),
     "ru": ("Сколько людей будет жить на Земле в 2150 году?", "Точно этого никто не знает. Демографические прогнозы на такой срок очень неопределённы и зависят от рождаемости, медицины и событий, которые мы не можем предсказать. Поэтому я не назову конкретную цифру как факт — могу рассказать, как делают такие прогнозы."),
     "en": ("How many people will live on Earth in 2150?", "Nobody knows for sure. Demographic projections that far ahead are highly uncertain and depend on birth rates, medicine and events we can't predict. So I won't state a specific number as fact — I can explain how such projections are made.")},
    {"uk": ("Що написано в моєму файлі на робочому столі?", "Я не бачу твоїх файлів і не маю до них доступу. Якщо вставиш сюди текст файлу, я з радістю його проаналізую."),
     "ru": ("Что написано в моём файле на рабочем столе?", "Я не вижу твоих файлов и не имею к ним доступа. Если вставишь сюда текст файла, с радостью его проанализирую."),
     "en": ("What does the file on my desktop say?", "I can't see your files and have no access to them. If you paste the text here, I'll gladly analyze it.")},
    {"uk": ("Розкажи про книгу «Зелений сніг над Лімою» автора Петра Нікольського.", "Я не знаю такої книги, і не хочу вигадувати її сюжет. Можливо, назва чи автор трохи інші, або це маловідомий твір. Якщо розкажеш, де про неї почув, чи дасиш уривок, спробую допомогти."),
     "ru": ("Расскажи о книге «Зелёный снег над Лимой» автора Петра Никольского.", "Я не знаю такой книги и не хочу выдумывать её сюжет. Возможно, название или автор немного другие, либо это малоизвестное произведение. Если расскажешь, где о ней услышал, или дашь отрывок, постараюсь помочь."),
     "en": ("Tell me about the book \"Green Snow Over Lima\" by Peter Nicholson.", "I don't recognise that book and I don't want to invent a plot. The title or author may be slightly different, or it may be an obscure work. If you tell me where you heard about it or share an excerpt, I'll try to help.")},
]

# ------------------------------------------------------------------ logic puzzles (parametric names)
NAMES = {
    "uk": ["Олена", "Андрій", "Марія", "Тарас", "Ірина", "Богдан", "Софія", "Максим", "Дарина", "Роман"],
    "ru": ["Елена", "Андрей", "Мария", "Тарас", "Ирина", "Борис", "София", "Максим", "Дарья", "Роман"],
    "en": ["Emma", "Liam", "Olivia", "Noah", "Ava", "Lucas", "Mia", "Ethan", "Chloe", "Owen"],
}
