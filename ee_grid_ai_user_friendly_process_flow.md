# EE Grid AI — User-Friendly Process Flow & UX Improvement Plan

**Project:** EE Grid AI  
**Document Type:** UX / Product Flow Recommendation  
**Based on:** EE Grid AI Release 1.0 SRS  
**Prepared for:** Product, UI/UX, and Development Planning  

---

## 1. Purpose of This Document

This document analyzes the current EE Grid AI Release 1.0 product flow and suggests a more user-friendly process flow for students.

The current SRS is technically strong and service-oriented, but the user journey should be simplified so that a student can quickly understand the platform, select a course, and start learning with AI.

The main UX goal should be:

> **Choose what I study → Ask AI → Upload material only when needed.**

---

## 2. Key UX Problem in the Current Flow

The current SRS focuses heavily on technical modules such as:

- Authentication
- Subscription and token billing
- Course service
- Document upload
- Document processing
- RAG pipeline
- Query optimization
- AI provider abstraction

These are important for development, but from a student’s point of view, the platform should not feel technical.

A student should not first think about:

- Document labels
- Processing status
- Token ledger
- Embeddings
- RAG scope
- Vector search
- Background queues

Instead, the student should immediately understand:

- What can I study?
- What course am I in?
- Can I ask AI questions?
- Can I upload my notes?
- Can I get exam help?

---

## 3. Recommended Main Student Flow

```text
Landing Page
   ↓
Sign up / Google Login
   ↓
Quick Setup
   - Choose study area
   - Choose category
   - Choose course
   - Choose level
   - Choose language
   ↓
Student Dashboard
   ↓
Main Actions
   1. Ask AI Tutor
   2. Browse Course Materials
   3. Upload My Notes / Book
   4. Study Topics
   ↓
AI Tutor
   - Ask from selected course
   - Ask from one document
   - Ask from all enrolled courses
   ↓
Answer Page
   - Simple answer
   - Step-by-step explanation
   - Source citation
   - Related topics
   - Follow-up suggestions
```

---

## 4. Most Important UX Principle

The biggest UX decision should be:

> **Do not make the user upload first. Let the user study first.**

The platform should feel like:

> “I choose my course and immediately ask questions.”

It should not feel like:

> “I need to upload, label, wait, process, and then maybe ask.”

Document upload is powerful, but it should be an optional enhancement, not the first barrier.

---

## 5. Recommended Onboarding Flow

After registration, the user should not go directly to a blank dashboard.

### Current Flow

```text
Register → Dashboard
```

### Recommended Flow

```text
Register
   ↓
Quick Setup
   ↓
Dashboard
```

### Quick Setup Steps

```text
Step 1: What are you studying?
- Engineering
- IELTS
- SSC / HSC
- Job Preparation

Step 2: Choose your category
- EEE
- Civil Engineering
- CSE
- IELTS
- SSC
- HSC

Step 3: Choose your course
- Circuit Theory
- Power Systems
- Electronics
- Control Systems

Step 4: Choose your level
- Beginner
- Intermediate
- Advanced

Step 5: Choose explanation language
- English
- Bangla
- Mixed
```

### Why This Helps

This makes the platform feel personalized from the beginning. The user immediately understands that the system is built around their study goal.

---

## 6. Recommended Dashboard Flow

The dashboard should be action-based, not data-based.

### Recommended Dashboard Layout

```text
Welcome back

Continue Studying
- Circuit Theory
- Power Systems

Main Actions
[Ask AI Tutor]
[Upload Study Material]
[Browse Books]
[View Topics]

Your Progress
- 3 courses enrolled
- 2 documents processed
- 14 topics available
```

### Dashboard Priorities

The dashboard should prioritize:

1. Ask AI Tutor
2. Continue studying
3. Browse course materials
4. Upload documents
5. View study plan

The dashboard should avoid showing technical details too early, such as:

- Token ledger details
- Processing backend details
- Embedding status
- API-level status
- RAG configuration

---

## 7. Recommended AI Tutor Flow

The AI Tutor should be the primary feature of the platform.

### AI Tutor Flow

```text
AI Tutor
   ↓
Select context
   - Current Course
   - Specific Book / Note
   - All My Courses
   ↓
Ask question
   ↓
AI answer
   ↓
Follow-up actions
```

### AI Answer Page Should Include

```text
Direct Answer
Step-by-Step Explanation
Source Citation
Related Topics
Follow-up Buttons
```

### Recommended Follow-Up Buttons

```text
[Explain Simply]
[Explain in Bangla]
[Give Formula]
[Solve Numerical]
[Generate Exam Questions]
[Make Short Notes]
[Give Viva Questions]
[Summarize This Topic]
```

### Why This Helps

Students often do not know how to write perfect prompts. These buttons guide them and make the AI easier to use.

---

## 8. Recommended Document Upload Flow

The current upload requirements are valid, but the UI should simplify them.

### Recommended Upload Flow

```text
Upload Material
   ↓
Step 1: Choose course
   ↓
Step 2: Upload file
   ↓
Step 3: Select material type
   - Book
   - Lecture Note
   - Previous Question
   - Other
   ↓
Step 4: Choose visibility
   - Private: only me
   - Public: share with course members
   ↓
Step 5: Confirm metadata
   ↓
Processing screen
```

### For Book Uploads

Instead of forcing the user to manually enter all details first, the system should try to detect metadata.

```text
We found:

Title: Electrical Circuit Theory
Author: Robert Boylestad

Is this correct?
[Yes] [Edit]
```

### Why This Helps

This reduces friction and makes the upload process feel smarter.

---

## 9. Recommended Processing Status Flow

The SRS already defines document status:

```text
pending → processing → processed / failed
```

But the user should see a friendly progress screen.

### Recommended Processing Screen

```text
Your file is being prepared for AI study

Status:
✓ File uploaded
✓ Text extracted
⏳ Creating searchable notes
⏳ Extracting topics
⏳ Ready for AI chat

You can leave this page. We will notify you when it is ready.
```

### After Processing Completes

```text
Your document is ready.

What do you want to do?
[Ask questions from this document]
[View extracted topics]
[Generate summary]
[Create exam questions]
```

### If Processing Fails

```text
We could not process this file properly.

Possible reasons:
- File is scanned
- Text is unclear
- File format is not supported
- OCR quality is low

Actions:
[Try again]
[Upload another file]
[Contact support]
```

---

## 10. Recommended Topic / Study Plan Flow

Topic extraction should not feel like a technical feature. It should feel like a study plan.

### Recommended Topic Page

```text
Study Plan from Your Document

Recommended order:
1. Basic Circuit Elements
2. Ohm’s Law
3. Kirchhoff’s Current Law
4. Kirchhoff’s Voltage Law
5. Nodal Analysis
6. Mesh Analysis
```

### Each Topic Should Have Actions

```text
Topic: Kirchhoff’s Voltage Law

[Study]
[Ask AI]
[Generate Questions]
[Make Short Notes]
[Mark as Done]
```

### Why This Helps

This turns extracted topics into an actual learning path.

---

## 11. Recommended Library Flow

Students should be able to browse public course materials without uploading anything.

### Library Flow

```text
Library
   ↓
Select Course
   ↓
View Materials
   - Public books
   - Previous questions
   - Lecture notes
   ↓
Open Material
   ↓
Actions
   - Ask from this book
   - View topics
   - Generate summary
   - Add to my library
```

### Why This Helps

It gives immediate value, especially for new users who do not have files ready.

---

## 12. Recommended Subscription / Token Flow

Token balance should be visible but not scary.

### Avoid Showing This First

```text
Token Ledger
Monthly Quota
Rate Limit
Usage Log
Billing Cycle
```

### Better Student-Friendly Version

```text
AI Usage

Plan: Free
Questions left this month: 35
Reset date: 1 June 2026

[Upgrade Plan]
```

### When Quota Is Exhausted

```text
You have used your free AI questions for this month.

You can:
[Upgrade Plan]
[Wait until reset date]
[Use non-AI study materials]
```

### Why This Helps

Students understand “questions left” more easily than “tokens used”.

---

## 13. Recommended Navigation Structure

### Desktop Sidebar

```text
Dashboard
AI Tutor
My Courses
Library
Uploads
Study Plan
Chat History
Subscription
Profile
```

### Future Mobile Navigation

```text
Home | AI Tutor | Courses | Uploads | Profile
```

---

## 14. Improved Complete User Flow

```text
1. Landing Page
   ↓
2. Sign up / Login
   ↓
3. Quick Setup
   - Select domain
   - Select category
   - Select course
   - Select level
   - Select language
   ↓
4. Dashboard
   - Continue course
   - Ask AI Tutor
   - Upload material
   - Browse books
   ↓
5A. Ask AI Tutor
   - Select scope: course / document / all courses
   - Ask question
   - Get answer with citation
   - Follow-up prompts
   ↓
5B. Upload Material
   - Select course
   - Upload file
   - Choose type
   - Confirm metadata
   - Processing status
   - Start studying
   ↓
5C. Browse Course Materials
   - Public books
   - Previous questions
   - Lecture notes
   - Add to my library
   ↓
6. Study Topic Page
   - Recommended order
   - Topic explanation
   - Related document pages
   - Practice questions
   ↓
7. Account / Subscription
   - Usage balance
   - Current plan
   - Upgrade only when needed
```

---

## 15. Suggested SRS Additions

Add a new section before Functional Requirements.

# User Experience Flow Requirements

| ID | Requirement | Priority |
|---|---|---|
| UX-01 | New users must complete a quick setup flow after registration | Must |
| UX-02 | User must select domain, category, course, level, and language preference during onboarding | Must |
| UX-03 | Dashboard must show primary actions: Ask AI Tutor, Upload Material, Browse Books, Study Topics | Must |
| UX-04 | User must be able to ask AI questions from seeded course materials without uploading first | Must |
| UX-05 | Upload flow must be step-by-step with clear progress and confirmation | Must |
| UX-06 | Document processing status must be visible in real time or near-real time | Must |
| UX-07 | After document processing, user must be guided to Ask AI, View Topics, or Generate Summary | Must |
| UX-08 | AI answer page must include follow-up action buttons such as Explain Simply, Give Example, Generate Exam Questions | Should |
| UX-09 | Topic page must show recommended study order and action buttons per topic | Should |
| UX-10 | Token balance should be visible but should not dominate the learning interface | Should |

---

## 16. Suggested Product Screens for Release 1

Minimum screens needed for a user-friendly Release 1:

1. Landing Page
2. Register / Login
3. Quick Setup
4. Dashboard
5. Course Catalogue
6. Course Detail Page
7. AI Tutor Page
8. AI Answer Page
9. Upload Material Page
10. Upload Processing Status Page
11. Library / Course Materials Page
12. Study Plan / Topics Page
13. Chat History Page
14. Subscription Page
15. Profile Settings Page

---

## 17. What to Change First

### Priority 1: Add Quick Setup

This should be added immediately because it defines the user’s study context.

### Priority 2: Make AI Tutor the Main CTA

The main dashboard button should be:

```text
Ask AI Tutor
```

not:

```text
Upload Document
```

### Priority 3: Allow AI Chat from Seeded Materials

Users should get value even before uploading files.

### Priority 4: Simplify Upload

Make upload step-by-step and friendly.

### Priority 5: Convert Topics into Study Plan

Extracted topics should become a guided learning path.

---

## 18. Final Recommendation

The product should be designed around the student’s goal, not around the system architecture.

### Best Release 1 Experience

```text
I sign up.
I choose EEE and my courses.
I ask AI questions immediately.
I upload my notes if I want better personal answers.
The system creates a study plan for me.
I prepare for exams faster.
```

That is the most user-friendly direction for EE Grid AI Release 1.
