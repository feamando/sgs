# G0 answer cards (heuristic)

### ch01 (chat)
**user:** When was IKEA founded?  
_(asked: When was IKEA founded? · year · NEW)_  
OK: Not sure, but possibly **1943**.
> IKEA was founded in 1943 by Ingvar Kamprad , and has been the world's largest furniture retailer since 2008.
Source: en.wikipedia.org · confidence 40%

**user:** And H&M?  
_(asked: When was H&M founded? · year · SWAP_ENTITY)_  
OK: **1947**.
> H&M was founded by Erling Persson in 1947 under the name Hennes.
Source: en.wikipedia.org · confidence 100%

**user:** What about Zara?  
_(asked: When was Zara founded? · year · SWAP_ENTITY)_  
abstain: I couldn't find an answer I'd trust for that. (searched for: "When was Zara founded?")


### c01 Compare the founding years of IKEA, H&M and Zara

| item | founded |
|---|---|
| IKEA | 1943 (en.wikipedia.org) |
| H&M | 1947 (en.wikipedia.org) |
| Zara | 2020 (en.wikipedia.org) |

### f01 (OK)

**What year was IKEA founded?**
  **1943**  (confidence 40%, single source, from memory)
  "IKEA was founded in 1943 by Ingvar Kamprad , and has been the world's largest furniture retailer since 2008."
  Source: https://en.wikipedia.org/wiki/IKEA (en.wikipedia.org)

### ch02 (chat)
**user:** Who founded SpaceX?  
_(asked: Who founded SpaceX? · entity · NEW)_  
OK: **Musk**.
> In February 2026 it became a subsidiary of spaceflight company SpaceX , also founded by Musk.
Source: en.wikipedia.org · confidence 87%

**user:** When was it founded?  
_(asked: When was SpaceX founded? · year · PRONOUN)_  
WRONG: **2026**.
> In February 2026 it became a subsidiary of spaceflight company SpaceX , also founded by Musk.
Source: en.wikipedia.org · confidence 96%


### c02 Compare the heights of the Burj Khalifa, Tokyo Skytree and the Eiffel Tower

| item | height |
|---|---|
| Burj Khalifa | 180 metres (en.wikipedia.org) |
| Tokyo Skytree | 634 metres (en.wikipedia.org) |
| Eiffel Tower | 158 metres (en.wikipedia.org) |

### f02 (OK)

**What year was LEGO founded?**
  **1932**  (confidence 90%, single source)
  "The Lego company was founded in 1932 by Ole Kirk Christiansen ,  a carpenter whose primary business of producing household goods had suffered due to the Great Depression ."
  Source: https://en.wikipedia.org/wiki/The_Lego_Group (en.wikipedia.org)

### ch03 (chat)
**user:** How tall is the Burj Khalifa in metres?  
_(asked: How tall is the Burj Khalifa in metres? · number · NEW)_  
WRONG: **180 metres** (from memory).
> Jeddah Tower is planned to be the first 1-kilometre-tall (1,000   m; 3,300   ft) building and, upon completion, would become the world's tallest building or structure , standing at least 180 metres (590   ft) taller than
Source: en.wikipedia.org · confidence 90%

**user:** and Tokyo Skytree?  
_(asked: How tall is the Tokyo Skytree in metres? · number · SWAP_ENTITY)_  
OK: Not sure, but possibly **634 metres** (from memory).
> It has been the tallest tower in Japan since opening in 2012,  and reached its full height of 634 metres (2,080 feet) in early 2011, making it the tallest tower in the world, displacing the Canton Tower ,   and the third
Source: en.wikipedia.org · confidence 40%


### c03 Compare the atomic numbers of gold, iron and oxygen

| item | atomic number |
|---|---|
| gold | 79 (en.wikipedia.org) |
| iron | 60 (en.wikipedia.org) |
| oxygen | 8 (en.wikipedia.org) |

### f03 (OK)

**What year was Nintendo founded?**
  **1889**  (confidence 66%, single source)
  "The history of Nintendo began when craftsman Fusajiro Yamauchi founded the company in 1889 to produce handmade hanafuda playing cards."
  Source: https://en.wikipedia.org/wiki/Nintendo (en.wikipedia.org)

### ch04 (chat)
**user:** What is the capital of Australia?  
_(asked: What is the capital of Australia? · entity · NEW)_  
WRONG: **Australian Capital Territory**.
> The Australian Capital Territory ( ACT ), known as the Federal Capital Territory until 1938, is an internal territory of Australia.
Source: en.wikipedia.org · confidence 79%

**user:** And Canada?  
_(asked: What is the capital of Canada? · entity · SWAP_ENTITY)_  
WRONG: Not sure, but possibly **Wikipedia**.
> Capital punishment in Canada - Wikipedia
Source: en.wikipedia.org · confidence 43%

**user:** What about Brazil?  
_(asked: What is the capital of Brazil? · entity · SWAP_ENTITY)_  
WRONG: Probably **Brazilian Portuguese**.
> Brasília ( / b r ə ˈ z ɪ l i ə / brə- ZIL -ee-ə ;   Brazilian Portuguese: [ bɾaˈziliɐ, bɾaˈziljɐ ] ⓘ ) is the capital city of Brazil and the Federal District .
Source: en.wikipedia.org · confidence 70%


### c04 Compare the release years of the PlayStation 5, Xbox Series X and Nintendo Switch

| item | released |
|---|---|
| PlayStation 5 | 2019 (en.wikipedia.org) |
| Xbox Series X | 2021 (en.wikipedia.org) |
| Nintendo Switch | 2025 (en.wikipedia.org) |

### f04 (abstain)

**What year was Sony founded?**
  I couldn't verify an answer (abstain).
