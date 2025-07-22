# Complete HTML & CSS Guide - Beginner to Advanced

## Table of Contents
1. [HTML Fundamentals](#html-fundamentals)
2. [CSS Fundamentals](#css-fundamentals)
3. [Intermediate HTML & CSS](#intermediate-html--css)
4. [Advanced CSS Concepts](#advanced-css-concepts)
5. [Modern CSS Features](#modern-css-features)
6. [Responsive Design](#responsive-design)
7. [CSS Architecture & Best Practices](#css-architecture--best-practices)
8. [Performance Optimization](#performance-optimization)
9. [Practical Projects](#practical-projects)
10. [Resources & Next Steps](#resources--next-steps)

---

## HTML Fundamentals

### What is HTML?
HTML (HyperText Markup Language) is the standard markup language for creating web pages. It describes the structure and content of web documents using elements and tags.

### Basic HTML Structure
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Page Title</title>
</head>
<body>
    <h1>Hello World!</h1>
</body>
</html>
```

### Essential HTML Elements

#### Text Elements
```html
<!-- Headings -->
<h1>Main Heading</h1>
<h2>Subheading</h2>
<h3>Sub-subheading</h3>
<!-- ... h4, h5, h6 -->

<!-- Paragraphs -->
<p>This is a paragraph of text.</p>

<!-- Text formatting -->
<strong>Bold text</strong>
<em>Italic text</em>
<u>Underlined text</u>
<mark>Highlighted text</mark>
<small>Small text</small>
<del>Deleted text</del>
<ins>Inserted text</ins>

<!-- Line breaks -->
<br> <!-- Line break -->
<hr> <!-- Horizontal rule -->
```

#### Lists
```html
<!-- Unordered list -->
<ul>
    <li>Item 1</li>
    <li>Item 2</li>
    <li>Item 3</li>
</ul>

<!-- Ordered list -->
<ol>
    <li>First item</li>
    <li>Second item</li>
    <li>Third item</li>
</ol>

<!-- Definition list -->
<dl>
    <dt>HTML</dt>
    <dd>HyperText Markup Language</dd>
    <dt>CSS</dt>
    <dd>Cascading Style Sheets</dd>
</dl>
```

#### Links and Images
```html
<!-- Links -->
<a href="https://example.com">External link</a>
<a href="#section">Internal link</a>
<a href="mailto:email@example.com">Email link</a>
<a href="tel:+1234567890">Phone link</a>

<!-- Images -->
<img src="image.jpg" alt="Description of image" width="300" height="200">

<!-- Figure with caption -->
<figure>
    <img src="image.jpg" alt="Description">
    <figcaption>Image caption</figcaption>
</figure>
```

#### Tables
```html
<table>
    <thead>
        <tr>
            <th>Header 1</th>
            <th>Header 2</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Data 1</td>
            <td>Data 2</td>
        </tr>
    </tbody>
</table>
```

#### Forms
```html
<form action="/submit" method="POST">
    <!-- Text inputs -->
    <input type="text" name="username" placeholder="Username" required>
    <input type="email" name="email" placeholder="Email" required>
    <input type="password" name="password" placeholder="Password" required>
    
    <!-- Other input types -->
    <input type="number" name="age" min="18" max="100">
    <input type="date" name="birthdate">
    <input type="checkbox" name="subscribe" id="subscribe">
    <label for="subscribe">Subscribe to newsletter</label>
    
    <!-- Radio buttons -->
    <input type="radio" name="gender" value="male" id="male">
    <label for="male">Male</label>
    <input type="radio" name="gender" value="female" id="female">
    <label for="female">Female</label>
    
    <!-- Select dropdown -->
    <select name="country">
        <option value="us">United States</option>
        <option value="uk">United Kingdom</option>
        <option value="ca">Canada</option>
    </select>
    
    <!-- Textarea -->
    <textarea name="message" rows="4" cols="50" placeholder="Your message"></textarea>
    
    <!-- Submit button -->
    <button type="submit">Submit</button>
</form>
```

### Semantic HTML5 Elements
```html
<header>
    <nav>
        <ul>
            <li><a href="home">Home</a></li>
            <li><a href="about">About</a></li>
        </ul>
    </nav>
</header>

<main>
    <article>
        <header>
            <h1>Article Title</h1>
            <time datetime="2024-01-01">January 1, 2024</time>
        </header>
        <section>
            <h2>Section Title</h2>
            <p>Article content...</p>
        </section>
    </article>
    
    <aside>
        <h3>Related Links</h3>
        <ul>
            <li><a href="#">Link 1</a></li>
            <li><a href="#">Link 2</a></li>
        </ul>
    </aside>
</main>

<footer>
    <p>&copy; 2024 Your Website</p>
</footer>
```

---

## CSS Fundamentals

### What is CSS?
CSS (Cascading Style Sheets) is used to style and layout web pages. It controls the visual presentation of HTML elements.

### CSS Syntax
```css
selector {
    property: value;
    property: value;
}
```

### Ways to Add CSS
```html
<!-- External CSS (Recommended) -->
<link rel="stylesheet" href="styles.css">

<!-- Internal CSS -->
<style>
    body { background-color: lightblue; }
</style>

<!-- Inline CSS (Not recommended) -->
<p style="color: red;">Red text</p>
```

### CSS Selectors

#### Basic Selectors
```css
/* Element selector */
p { color: blue; }

/* Class selector */
.highlight { background-color: yellow; }

/* ID selector */
#header { font-size: 24px; }

/* Universal selector */
* { margin: 0; padding: 0; }
```

#### Combinator Selectors
```css
/* Descendant selector */
div p { color: red; }

/* Child selector */
div > p { color: green; }

/* Adjacent sibling selector */
h1 + p { margin-top: 0; }

/* General sibling selector */
h1 ~ p { color: gray; }
```

#### Attribute Selectors
```css
/* Has attribute */
[title] { cursor: help; }

/* Exact attribute value */
[type="text"] { border: 1px solid #ccc; }

/* Attribute contains value */
[class*="btn"] { padding: 10px; }

/* Attribute starts with value */
[href^="https"] { color: green; }

/* Attribute ends with value */
[href$=".pdf"] { color: red; }
```

#### Pseudo-classes
```css
/* Link states */
a:link { color: blue; }
a:visited { color: purple; }
a:hover { color: red; }
a:active { color: orange; }

/* Form states */
input:focus { border: 2px solid blue; }
input:disabled { opacity: 0.5; }
input:checked + label { font-weight: bold; }

/* Structural pseudo-classes */
p:first-child { font-weight: bold; }
p:last-child { margin-bottom: 0; }
tr:nth-child(even) { background-color: #f2f2f2; }
tr:nth-child(odd) { background-color: white; }
```

#### Pseudo-elements
```css
/* First line/letter */
p::first-line { font-weight: bold; }
p::first-letter { font-size: 2em; }

/* Before/after content */
.quote::before { content: '"'; }
.quote::after { content: '"'; }

/* Placeholder styling */
input::placeholder { color: #999; }
```

### CSS Properties

#### Text and Font Properties
```css
.text-styling {
    /* Font properties */
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 16px;
    font-weight: bold; /* or 700 */
    font-style: italic;
    line-height: 1.5;
    
    /* Text properties */
    color: #333;
    text-align: center;
    text-decoration: underline;
    text-transform: uppercase;
    letter-spacing: 1px;
    word-spacing: 2px;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
}
```

#### Background Properties
```css
.background-styling {
    background-color: #f0f0f0;
    background-image: url('image.jpg');
    background-repeat: no-repeat;
    background-position: center center;
    background-size: cover;
    background-attachment: fixed;
    
    /* Shorthand */
    background: #f0f0f0 url('image.jpg') no-repeat center/cover fixed;
}
```

#### Border Properties
```css
.border-styling {
    border-width: 1px;
    border-style: solid;
    border-color: #ccc;
    
    /* Shorthand */
    border: 1px solid #ccc;
    
    /* Individual sides */
    border-top: 2px solid red;
    border-right: 1px dashed blue;
    border-bottom: 3px dotted green;
    border-left: 1px solid black;
    
    /* Border radius */
    border-radius: 10px;
    border-radius: 10px 20px; /* top-left/bottom-right, top-right/bottom-left */
    border-radius: 10px 20px 30px 40px; /* top-left, top-right, bottom-right, bottom-left */
}
```

### The CSS Box Model
```css
.box-model {
    /* Content area */
    width: 200px;
    height: 100px;
    
    /* Padding (inside border) */
    padding: 20px;
    padding-top: 10px;
    padding-right: 15px;
    padding-bottom: 20px;
    padding-left: 25px;
    
    /* Border */
    border: 2px solid #ccc;
    
    /* Margin (outside border) */
    margin: 10px;
    margin-top: 5px;
    margin-right: auto; /* Centering technique */
    margin-bottom: 15px;
    margin-left: auto; /* Centering technique */
    
    /* Box sizing */
    box-sizing: border-box; /* Includes padding and border in width/height */
}
```

### Display Property
```css
/* Block elements */
.block { display: block; }

/* Inline elements */
.inline { display: inline; }

/* Inline-block elements */
.inline-block { display: inline-block; }

/* Hide elements */
.hidden { display: none; }

/* Table display */
.table { display: table; }
.table-row { display: table-row; }
.table-cell { display: table-cell; }
```

### Position Property
```css
/* Static (default) */
.static { position: static; }

/* Relative positioning */
.relative {
    position: relative;
    top: 10px;
    left: 20px;
}

/* Absolute positioning */
.absolute {
    position: absolute;
    top: 50px;
    right: 100px;
}

/* Fixed positioning */
.fixed {
    position: fixed;
    bottom: 20px;
    right: 20px;
}

/* Sticky positioning */
.sticky {
    position: sticky;
    top: 0;
}
```

---

## Intermediate HTML & CSS

### Advanced CSS Selectors and Techniques

#### Advanced Pseudo-classes
```css
/* Negation pseudo-class */
p:not(.exclude) { color: blue; }

/* Root pseudo-class */
:root {
    --main-color: #3498db;
    --secondary-color: #e74c3c;
}

/* Target pseudo-class */
:target { background-color: yellow; }

/* Empty pseudo-class */
p:empty { display: none; }

/* Language pseudo-class */
:lang(en) { quotes: '"' '"'; }
```

#### Custom Properties (CSS Variables)
```css
:root {
    --primary-color: #3498db;
    --secondary-color: #e74c3c;
    --font-size-large: 24px;
    --spacing-unit: 16px;
}

.component {
    color: var(--primary-color);
    font-size: var(--font-size-large);
    margin: calc(var(--spacing-unit) * 2);
    
    /* Fallback value */
    background: var(--undefined-color, #f0f0f0);
}

/* Dynamic variables with JavaScript */
.theme-dark {
    --primary-color: #2c3e50;
    --text-color: white;
}
```

### Flexbox Layout
```css
/* Flex container */
.container {
    display: flex;
    
    /* Direction */
    flex-direction: row; /* row | row-reverse | column | column-reverse */
    
    /* Wrapping */
    flex-wrap: wrap; /* nowrap | wrap | wrap-reverse */
    
    /* Shorthand for direction and wrap */
    flex-flow: row wrap;
    
    /* Main axis alignment */
    justify-content: space-between; /* flex-start | flex-end | center | space-between | space-around | space-evenly */
    
    /* Cross axis alignment */
    align-items: center; /* stretch | flex-start | flex-end | center | baseline */
    
    /* Multiple line alignment */
    align-content: space-around; /* stretch | flex-start | flex-end | center | space-between | space-around */
    
    /* Gap between items */
    gap: 20px;
    row-gap: 10px;
    column-gap: 20px;
}

/* Flex items */
.item {
    /* Growth factor */
    flex-grow: 1;
    
    /* Shrink factor */
    flex-shrink: 0;
    
    /* Base size */
    flex-basis: 200px;
    
    /* Shorthand */
    flex: 1 0 200px; /* grow shrink basis */
    
    /* Individual alignment */
    align-self: flex-end;
    
    /* Order */
    order: 2;
}
```

### CSS Grid Layout
```css
/* Grid container */
.grid-container {
    display: grid;
    
    /* Define columns and rows */
    grid-template-columns: 1fr 2fr 1fr;
    grid-template-rows: 100px 1fr 50px;
    
    /* Using repeat() */
    grid-template-columns: repeat(3, 1fr);
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    
    /* Named grid lines */
    grid-template-columns: [sidebar-start] 250px [sidebar-end main-start] 1fr [main-end];
    
    /* Grid areas */
    grid-template-areas: 
        "header header header"
        "sidebar main main"
        "footer footer footer";
    
    /* Gaps */
    grid-gap: 20px;
    grid-row-gap: 10px;
    grid-column-gap: 20px;
    
    /* Alignment */
    justify-items: center; /* start | end | center | stretch */
    align-items: center; /* start | end | center | stretch */
    justify-content: space-between; /* start | end | center | stretch | space-around | space-between | space-evenly */
    align-content: center; /* start | end | center | stretch | space-around | space-between | space-evenly */
}

/* Grid items */
.grid-item {
    /* Positioning by line numbers */
    grid-column-start: 1;
    grid-column-end: 3;
    grid-row-start: 2;
    grid-row-end: 4;
    
    /* Shorthand */
    grid-column: 1 / 3;
    grid-row: 2 / 4;
    grid-area: 2 / 1 / 4 / 3; /* row-start / column-start / row-end / column-end */
    
    /* Span notation */
    grid-column: span 2;
    grid-row: span 3;
    
    /* Named areas */
    grid-area: header;
    
    /* Individual alignment */
    justify-self: end;
    align-self: center;
}
```

### Transforms and Transitions
```css
/* 2D Transforms */
.transform-2d {
    transform: translate(50px, 100px);
    transform: rotate(45deg);
    transform: scale(1.5);
    transform: skew(20deg, 10deg);
    
    /* Combine transforms */
    transform: translate(50px, 100px) rotate(45deg) scale(1.2);
    
    /* Transform origin */
    transform-origin: top left;
    transform-origin: 50% 50%; /* center (default) */
}

/* 3D Transforms */
.transform-3d {
    transform-style: preserve-3d;
    perspective: 1000px;
    
    transform: translateZ(50px);
    transform: rotateX(45deg);
    transform: rotateY(45deg);
    transform: rotateZ(45deg);
    transform: translate3d(50px, 100px, 25px);
    transform: rotate3d(1, 1, 0, 45deg);
}

/* Transitions */
.transition {
    transition-property: all;
    transition-duration: 0.3s;
    transition-timing-function: ease-in-out;
    transition-delay: 0.1s;
    
    /* Shorthand */
    transition: all 0.3s ease-in-out 0.1s;
    
    /* Multiple properties */
    transition: opacity 0.3s ease, transform 0.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}

.transition:hover {
    opacity: 0.7;
    transform: scale(1.1);
}
```

### CSS Animations
```css
/* Define keyframes */
@keyframes slideIn {
    0% {
        transform: translateX(-100%);
        opacity: 0;
    }
    50% {
        transform: translateX(-50%);
        opacity: 0.5;
    }
    100% {
        transform: translateX(0);
        opacity: 1;
    }
}

/* Alternative percentage notation */
@keyframes bounce {
    from { transform: translateY(0); }
    50% { transform: translateY(-50px); }
    to { transform: translateY(0); }
}

/* Apply animation */
.animated {
    animation-name: slideIn;
    animation-duration: 1s;
    animation-timing-function: ease-out;
    animation-delay: 0.5s;
    animation-iteration-count: infinite;
    animation-direction: alternate;
    animation-fill-mode: forwards;
    animation-play-state: running;
    
    /* Shorthand */
    animation: slideIn 1s ease-out 0.5s infinite alternate forwards;
    
    /* Multiple animations */
    animation: slideIn 1s ease-out, fadeIn 0.5s ease-in;
}
```

---

## Advanced CSS Concepts

### CSS Functions
```css
.functions {
    /* calc() function */
    width: calc(100% - 40px);
    margin: calc(1rem + 10px);
    
    /* min(), max(), clamp() */
    width: min(500px, 100%);
    font-size: max(16px, 1rem);
    font-size: clamp(16px, 2.5vw, 32px); /* min, preferred, max */
    
    /* Color functions */
    color: rgb(255, 0, 0);
    color: rgba(255, 0, 0, 0.5);
    color: hsl(0, 100%, 50%);
    color: hsla(0, 100%, 50%, 0.5);
    
    /* Gradient functions */
    background: linear-gradient(45deg, red, blue);
    background: radial-gradient(circle, red, blue);
    background: conic-gradient(red, yellow, green, blue, red);
    
    /* Filter functions */
    filter: blur(5px);
    filter: brightness(150%);
    filter: contrast(200%);
    filter: grayscale(100%);
    filter: hue-rotate(90deg);
    filter: saturate(200%);
    filter: sepia(100%);
    filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.5));
}
```

### CSS Pseudo-elements Advanced
```css
/* Complex content generation */
.tooltip::after {
    content: attr(data-tooltip);
    position: absolute;
    background: black;
    color: white;
    padding: 5px 10px;
    border-radius: 4px;
    opacity: 0;
    transition: opacity 0.3s;
}

.tooltip:hover::after {
    opacity: 1;
}

/* Counter styling */
.counter {
    counter-reset: section;
}

.counter h2::before {
    counter-increment: section;
    content: "Section " counter(section) ": ";
}

/* Multi-column layouts */
.columns {
    column-count: 3;
    column-gap: 20px;
    column-rule: 1px solid #ccc;
    column-fill: balance;
}

.columns h2 {
    column-span: all;
}
```

### Advanced Layout Techniques
```css
/* Container queries (modern browsers) */
@container (min-width: 500px) {
    .card {
        display: flex;
    }
}

/* CSS Shapes */
.shape {
    width: 200px;
    height: 200px;
    float: left;
    shape-outside: circle(50%);
    clip-path: circle(50%);
}

/* CSS Masks */
.masked {
    -webkit-mask: url(mask.svg);
    mask: url(mask.svg);
    -webkit-mask-size: cover;
    mask-size: cover;
}

/* Scroll behavior */
html {
    scroll-behavior: smooth;
}

.scroll-snap-container {
    scroll-snap-type: y mandatory;
    overflow-y: scroll;
    height: 100vh;
}

.scroll-snap-item {
    scroll-snap-align: start;
    height: 100vh;
}
```

---

## Modern CSS Features

### CSS Logical Properties
```css
.logical-properties {
    /* Instead of margin-left/right */
    margin-inline-start: 20px;
    margin-inline-end: 10px;
    margin-inline: 20px 10px;
    
    /* Instead of margin-top/bottom */
    margin-block-start: 15px;
    margin-block-end: 25px;
    margin-block: 15px 25px;
    
    /* Border logical properties */
    border-inline-start: 2px solid red;
    border-block-end: 1px solid blue;
    
    /* Size logical properties */
    inline-size: 300px; /* width in horizontal writing mode */
    block-size: 200px; /* height in horizontal writing mode */
}
```

### CSS Subgrid
```css
.grid-parent {
    display: grid;
    grid-template-columns: 1fr 2fr 1fr;
    grid-template-rows: repeat(3, 100px);
    gap: 20px;
}

.grid-child {
    display: grid;
    grid-column: 1 / -1;
    grid-template-columns: subgrid;
    grid-template-rows: subgrid;
}
```

### CSS Layers (Cascade Layers)
```css
/* Define layer order */
@layer reset, base, theme, components, utilities;

/* Layer-specific styles */
@layer reset {
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
}

@layer base {
    body {
        font-family: system-ui, sans-serif;
        line-height: 1.5;
    }
}

@layer components {
    .button {
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 4px;
        background: blue;
        color: white;
    }
}
```

### Modern Color Spaces
```css
.modern-colors {
    /* Display P3 color space */
    color: color(display-p3 1 0 0);
    
    /* Lab color space */
    color: lab(50% 20 -30);
    
    /* LCH color space */
    color: lch(50% 30 180);
    
    /* Relative color syntax */
    background: hsl(from var(--primary-color) h s calc(l * 0.8));
}
```

### CSS Nesting (Native)
```css
.card {
    background: white;
    border-radius: 8px;
    padding: 1rem;
    
    & .title {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        
        &:hover {
            color: blue;
        }
    }
    
    & .content {
        line-height: 1.6;
        
        & p {
            margin-bottom: 1rem;
            
            &:last-child {
                margin-bottom: 0;
            }
        }
    }
    
    &:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    @media (max-width: 768px) {
        padding: 0.5rem;
    }
}
```

---

## Responsive Design

### Mobile-First Approach
```css
/* Base styles for mobile */
.container {
    width: 100%;
    padding: 1rem;
}

.grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 1rem;
}

/* Tablet styles */
@media (min-width: 768px) {
    .container {
        max-width: 750px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    .grid {
        grid-template-columns: repeat(2, 1fr);
        gap: 2rem;
    }
}

/* Desktop styles */
@media (min-width: 1024px) {
    .container {
        max-width: 1200px;
        padding: 3rem;
    }
    
    .grid {
        grid-template-columns: repeat(3, 1fr);
        gap: 3rem;
    }
}

/* Large desktop */
@media (min-width: 1440px) {
    .container {
        max-width: 1400px;
    }
}
```

### Responsive Typography
```css
/* Fluid typography */
h1 {
    font-size: clamp(2rem, 5vw, 4rem);
}

p {
    font-size: clamp(1rem, 2.5vw, 1.2rem);
    line-height: 1.6;
}

/* Responsive spacing */
.section {
    padding: clamp(2rem, 5vw, 5rem) 0;
}
```

### CSS Container Queries
```css
.card-container {
    container-type: inline-size;
    container-name: card;
}

@container card (min-width: 300px) {
    .card {
        display: flex;
        align-items: center;
    }
    
    .card img {
        width: 100px;
        height: 100px;
        margin-right: 1rem;
    }
}

@container card (min-width: 500px) {
    .card {
        flex-direction: column;
    }
    
    .card img {
        width: 200px;
        height: 200px;
        margin: 0 0 1rem 0;
    }
}
```

### Responsive Images
```css
/* Responsive images */
img {
    max-width: 100%;
    height: auto;
}

/* Art direction with picture element */
picture {
    display: block;
}

/* CSS-only responsive images */
.hero-image {
    width: 100%;
    height: 50vh;
    background-image: url('hero-mobile.jpg');
    background-size: cover;
    background-position: center;
}

@media (min-width: 768px) {
    .hero-image {
        background-image: url('hero-tablet.jpg');
        height: 60vh;
    }
}

@media (min-width: 1024px) {
    .hero-image {
        background-image: url('hero-desktop.jpg');
        height: 80vh;
    }
}
```

---

## CSS Architecture & Best Practices

### CSS Methodologies

#### BEM (Block Element Modifier)
```css
/* Block */
.card {
    background: white;
    border-radius: 8px;
    padding: 1rem;
}

/* Element */
.card__header {
    border-bottom: 1px solid #eee;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

.card__title {
    font-size: 1.2rem;
    font-weight: bold;
}

.card__content {
    line-height: 1.6;
}

/* Modifier */
.card--featured {
    border: 2px solid gold;
}

.card--large {
    padding: 2rem;
}

.card__title--center {
    text-align: center;
}
```

#### OOCSS (Object-Oriented CSS)
```css
/* Structure */
.box {
    border: 1px solid;
    border-radius: 4px;
}

/* Skin */
.box-primary {
    border-color: blue;
    background: lightblue;
}

.box-warning {
    border-color: orange;
    background: lightyellow;
}

/* Size */
.box-small { padding: 0.5rem; }
.box-medium { padding: 1rem; }
.box-large { padding: 2rem; }
```

### CSS Custom Properties Organization
```css
:root {
    /* Color system */
    --color-primary-50: #eff6ff;
    --color-primary-100: #dbeafe;
    --color-primary-500: #3b82f6;
    --color-primary-900: #1e3a8a;
    
    /* Typography scale */
    --font-size-xs: 0.75rem;
    --font-size-sm: 0.875rem;
    --font-size-base: 1rem;
    --font-size-lg: 1.125rem;
    --font-size-xl: 1.25rem;
    --font-size-2xl: 1.5rem;
    --font-size-3xl: 1.875rem;
    
    /* Spacing scale */
    --spacing-1: 0.25rem;
    --spacing-2: 0.5rem;
    --spacing-3: 0.75rem;
    --spacing-4: 1rem;
    --spacing-6: 1.5rem;
    --spacing-8: 2rem;
    
    /* Border radius */
    --radius-sm: 0.125rem;
    --radius-base: 0.25rem;
    --radius-md: 0.375rem;
    --radius-lg: 0.5rem;
    --radius-full: 9999px;
    
    /* Shadows */
    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
    --shadow-base: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
    --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
}
```

### Performance Best Practices
```css
/* Efficient selectors */
.good-selector { } /* Good: Class selector */
#specific-id { } /* Good: ID selector */
article > p { } /* Good: Child selector */

/* Avoid these */
* { } /* Bad: Universal selector */
div div div p { } /* Bad: Deep nesting */
[data-attribute="value"] { } /* Slower: Attribute selector */

/* Hardware acceleration */
.accelerated {
    transform: translateZ(0); /* Forces hardware acceleration */
    will-change: transform; /* Hint to browser */
}

/* Reduce repaints and reflows */
.optimized {
    /* Use transform instead of changing position */
    transform: translateX(100px);
    
    /* Use opacity instead of visibility */
    opacity: 0;
    
    /* Batch DOM changes */
    transition: transform 0.3s, opacity 0.3s;
}
```

### CSS Reset and Normalize
```css
/* Modern CSS Reset */
*, *::before, *::after {
    box-sizing: border-box;
}

* {
    margin: 0;
}

html, body {
    height: 100%;
}

body {
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
}

img, picture, video, canvas, svg {
    display: block;
    max-width: 100%;
}

input, button, textarea, select {
    font: inherit;
}

p, h1, h2, h3, h4, h5, h6 {
    overflow-wrap: break-word;
}

#root, #__next {
    isolation: isolate;
}
```

---

## Performance Optimization

### Critical CSS
```html
<!-- Inline critical CSS -->
<style>
    /* Above-the-fold styles */
    body { font-family: system-ui; }
    .header { background: #fff; }
    .hero { height: 100vh; }
</style>

<!-- Load non-critical CSS asynchronously -->
<link rel="preload" href="styles.css" as="style" onload="this.onload=null;this.rel='stylesheet'">
<noscript><link rel="stylesheet" href="styles.css"></noscript>
```

### CSS Optimization Techniques
```css
/* Use efficient transforms */
.move-element {
    /* Good: GPU accelerated */
    transform: translate3d(10px, 0, 0);
    
    /* Bad: Causes layout */
    /* left: 10px; */
}

/* Optimize animations */
.optimized-animation {
    /* Only animate compositor properties */
    animation: slide 1s ease-out;
}

@keyframes slide {
    from {
        transform: translateX(-100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

/* Minimize selector complexity */
.simple-selector { color: blue; }

/* Instead of */
/* div.container > ul.list li.item a.link:hover { color: blue; } */
```

### Loading Optimization
```css
/* Font loading optimization */
@font-face {
    font-family: 'CustomFont';
    src: url('font.woff2') format('woff2');
    font-display: swap; /* Improves loading performance */
}

/* Image optimization */
.background-image {
    background-image: url('image.webp');
    background-image: image-set(
        url('image.webp') type('image/webp'),
        url('image.jpg') type('image/jpeg')
    );
}

/* Lazy loading with CSS */
img[loading="lazy"] {
    opacity: 0;
    transition: opacity 0.3s;
}

img[loading="lazy"].loaded {
    opacity: 1;
}
```

---

## Practical Projects

### Project 1: Responsive Navigation
```css
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 2rem;
    background: white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.nav-list {
    display: flex;
    list-style: none;
    gap: 2rem;
    margin: 0;
    padding: 0;
}

.nav-link {
    text-decoration: none;
    color: #333;
    transition: color 0.3s;
}

.nav-link:hover {
    color: #007bff;
}

.hamburger {
    display: none;
    flex-direction: column;
    cursor: pointer;
}

.hamburger span {
    width: 25px;
    height: 3px;
    background: #333;
    margin: 3px 0;
    transition: 0.3s;
}

@media (max-width: 768px) {
    .nav-list {
        position: fixed;
        top: 70px;
        left: -100%;
        width: 100%;
        height: calc(100vh - 70px);
        background: white;
        flex-direction: column;
        justify-content: start;
        align-items: center;
        padding-top: 2rem;
        transition: left 0.3s;
    }
    
    .nav-list.active {
        left: 0;
    }
    
    .hamburger {
        display: flex;
    }
    
    .hamburger.active span:nth-child(1) {
        transform: rotate(45deg) translate(5px, 5px);
    }
    
    .hamburger.active span:nth-child(2) {
        opacity: 0;
    }
    
    .hamburger.active span:nth-child(3) {
        transform: rotate(-45deg) translate(7px, -6px);
    }
}
```

### Project 2: Card Layout with Grid
```css
.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 2rem;
    padding: 2rem;
}

.card {
    background: white;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    transition: transform 0.3s, box-shadow 0.3s;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
}

.card__image {
    width: 100%;
    height: 200px;
    object-fit: cover;
}

.card__content {
    padding: 1.5rem;
}

.card__title {
    font-size: 1.25rem;
    font-weight: bold;
    margin-bottom: 0.5rem;
    color: #333;
}

.card__description {
    color: #666;
    line-height: 1.6;
    margin-bottom: 1rem;
}

.card__button {
    display: inline-block;
    padding: 0.5rem 1rem;
    background: #007bff;
    color: white;
    text-decoration: none;
    border-radius: 6px;
    transition: background-color 0.3s;
}

.card__button:hover {
    background: #0056b3;
}
```

### Project 3: CSS-Only Modal
```css
.modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0,0,0,0.5);
    display: flex;
    justify-content: center;
    align-items: center;
    opacity: 0;
    visibility: hidden;
    transition: opacity 0.3s, visibility 0.3s;
}

.modal-overlay:target {
    opacity: 1;
    visibility: visible;
}

.modal {
    background: white;
    border-radius: 8px;
    padding: 2rem;
    max-width: 500px;
    width: 90%;
    max-height: 80vh;
    overflow-y: auto;
    position: relative;
    transform: scale(0.8);
    transition: transform 0.3s;
}

.modal-overlay:target .modal {
    transform: scale(1);
}

.modal__close {
    position: absolute;
    top: 1rem;
    right: 1rem;
    width: 30px;
    height: 30px;
    border: none;
    background: #f0f0f0;
    border-radius: 50%;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    text-decoration: none;
    color: #666;
}

.modal__close:hover {
    background: #e0e0e0;
}
```

---

## Resources & Next Steps

### Essential Tools and Resources
1. **Development Tools**
   - Browser DevTools (Chrome, Firefox, Safari)
   - VS Code with extensions (Live Server, Prettier, CSS Peek)
   - CSS validators and linters

2. **CSS Frameworks to Explore**
   - Tailwind CSS (utility-first)
   - Bootstrap (component-based)
   - Bulma (modern CSS framework)

3. **CSS Preprocessors**
   - Sass/SCSS
   - Less
   - Stylus

4. **Online Resources**
   - MDN Web Docs
   - CSS-Tricks
   - Can I Use (browser compatibility)
   - CodePen (for experimentation)

5. **Performance Tools**
   - Lighthouse
   - WebPageTest
   - CSS analyzer tools

### Next Learning Steps
1. **JavaScript Integration**
   - Learn how CSS interacts with JavaScript
   - DOM manipulation and styling
   - CSS-in-JS solutions

2. **Advanced Frameworks**
   - CSS modules
   - Styled-components
   - Emotion

3. **Design Systems**
   - Creating scalable CSS architectures
   - Component libraries
   - Design tokens

4. **Accessibility**
   - Semantic HTML and CSS
   - WCAG guidelines
   - Screen reader considerations

5. **Modern Workflow**
   - CSS build tools (PostCSS, etc.)
   - Version control with Git
   - Deployment strategies

### Practice Projects
1. Build a complete responsive website
2. Create a CSS component library
3. Implement complex layouts (magazine-style, dashboard)
4. Build CSS animations and interactions
5. Create a dark/light theme system

### Advanced Topics to Explore
- CSS Houdini (CSS custom properties API)
- Web Components styling
- CSS containment
- Advanced CSS architecture patterns
- Performance optimization techniques
- CSS testing strategies

---

This guide covers the journey from HTML & CSS basics to advanced concepts. Practice regularly, build projects, and stay updated with the latest web standards to continue growing your skills!