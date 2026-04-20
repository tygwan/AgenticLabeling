# Dashboard UI/UX Design and Architecture

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                                                 │
│                  Web Browser                    │
│                                                 │
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│                                                 │
│                 Frontend Layer                  │
│                                                 │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐   │
│  │   Views   │  │   State   │  │ Components│   │
│  │(HTML/CSS) │  │ Management│  │  Library  │   │
│  └───────────┘  └───────────┘  └───────────┘   │
│                                                 │
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│                                                 │
│                  API Layer                      │
│                                                 │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐   │
│  │  REST     │  │ WebSocket │  │  Image    │   │
│  │ Endpoints │  │ Events    │  │ Processing│   │
│  └───────────┘  └───────────┘  └───────────┘   │
│                                                 │
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│                                                 │
│                 Service Layer                   │
│                                                 │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐   │
│  │Classification│ Metadata  │  │ File      │   │
│  │ Services   │  │ Services │  │ Operations│   │
│  └───────────┘  └───────────┘  └───────────┘   │
│                                                 │
└───────────────────────┬─────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│                                                 │
│                 Data Layer                      │
│                                                 │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐   │
│  │ Filesystem│  │ Metadata  │  │ Image     │   │
│  │ Storage   │  │ JSON      │  │ Files     │   │
│  └───────────┘  └───────────┘  └───────────┘   │
│                                                 │
└─────────────────────────────────────────────────┘
```

## Data Flow

1. **User Interactions**
   - User performs actions in the web interface
   - Frontend components capture these events and update UI state

2. **API Communication**
   - Frontend sends requests to backend API endpoints
   - REST API handles CRUD operations
   - WebSockets provide real-time updates for long-running operations

3. **Service Processing**
   - Backend services process requests
   - Classification logic applies user settings to images
   - Metadata services extract and manage image information
   - File operations handle moving/copying/organizing files

4. **Data Persistence**
   - Changes are persisted to filesystem
   - Metadata is stored in JSON files
   - Image files are organized according to classification results

## Wireframes

### 1. Dashboard Home

```
┌─────────────────────────────────────────────────────────────────────┐
│ AI Image Processing Dashboard                              [User ▼] │
├─────────────────────────────────────────────────────────────────────┤
│ │ Categories │ Support Set │ Refined Dataset │ Analytics │ Settings │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │
│  │ Category 1  │  │ Category 2  │  │ Category 3  │  │ Category 4  │ │
│  │             │  │             │  │             │  │             │ │
│  │ Items: 120  │  │ Items: 85   │  │ Items: 42   │  │ Items: 67   │ │
│  │ Classes: 5  │  │ Classes: 3  │  │ Classes: 4  │  │ Classes: 2  │ │
│  │             │  │             │  │             │  │             │ │
│  │ [Open]      │  │ [Open]      │  │ [Open]      │  │ [Open]      │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Recent Activity                                                 │ │
│  │                                                                 │ │
│  │ • Category 2: 15 images classified (Method 1) - 10 mins ago    │ │
│  │ • Category 1: 5 images reclassified - 1 hour ago               │ │
│  │ • Category 3: Support set updated - 3 hours ago                │ │
│  │                                                                 │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌───────────────────────────┐  ┌───────────────────────────────────┐ │
│  │ System Status             │  │ Classification Performance        │ │
│  │                           │  │                                   │ │
│  │ • CPU: 12%                │  │ Category 1: 92% accuracy          │ │
│  │ • Memory: 1.2GB/8GB       │  │ Category 2: 87% accuracy          │ │
│  │ • Storage: 34GB/500GB     │  │ Category 3: 95% accuracy          │ │
│  │ • Last backup: 2 days ago │  │ Category 4: 89% accuracy          │ │
│  │                           │  │                                   │ │
│  └───────────────────────────┘  └───────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2. Support Set Browser

```
┌─────────────────────────────────────────────────────────────────────┐
│ AI Image Processing Dashboard                              [User ▼] │
├─────────────────────────────────────────────────────────────────────┤
│ │ Categories │ Support Set │ Refined Dataset │ Analytics │ Settings │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────┐   ┌───────────────────────────────────────┐  │
│  │                   │   │                                       │  │
│  │  Class Navigation │   │           Image Gallery              │  │
│  │                   │   │                                       │  │
│  │  ● Class A (12)   │   │  ┌────────┐ ┌────────┐ ┌────────┐    │  │
│  │  ○ Class B (8)    │   │  │        │ │        │ │        │    │  │
│  │  ○ Class C (15)   │   │  │  Img 1 │ │  Img 2 │ │  Img 3 │    │  │
│  │  ○ Class D (5)    │   │  │        │ │        │ │        │    │  │
│  │                   │   │  └────────┘ └────────┘ └────────┘    │  │
│  │  [+ Add Class]    │   │                                       │  │
│  │                   │   │  ┌────────┐ ┌────────┐ ┌────────┐    │  │
│  │  Filters:         │   │  │        │ │        │ │        │    │  │
│  │  [Confidence▼]    │   │  │  Img 4 │ │  Img 5 │ │  Img 6 │    │  │
│  │  [Size▼]          │   │  │        │ │        │ │        │    │  │
│  │  [Date▼]          │   │  └────────┘ └────────┘ └────────┘    │  │
│  │                   │   │                                       │  │
│  └───────────────────┘   │  ┌────────┐ ┌────────┐ ┌────────┐    │  │
│                          │  │        │ │        │ │        │    │  │
│  ┌───────────────────┐   │  │  Img 7 │ │  Img 8 │ │  Img 9 │    │  │
│  │  Selected Image   │   │  │        │ │        │ │        │    │  │
│  │  ┌─────────────┐  │   │  └────────┘ └────────┘ └────────┘    │  │
│  │  │             │  │   │                                       │  │
│  │  │             │  │   │            [1] [2] [3] Next >         │  │
│  │  │   Preview   │  │   └───────────────────────────────────────┘  │
│  │  │             │  │                                               │
│  │  │             │  │   ┌───────────────────────────────────────┐  │
│  │  └─────────────┘  │   │ Actions:                              │  │
│  │                   │   │ [Upload New] [Delete] [Reclassify▼]   │  │
│  │  ID: img_12345    │   │                                       │  │
│  │  Size: 640x480    │   │ [Apply to Selected (3)] [Apply to All]   │
│  │  Date: 2023-05-12 │   └───────────────────────────────────────┘  │
│  │  Confidence: 87%  │                                               │
│  └───────────────────┘                                               │
└─────────────────────────────────────────────────────────────────────┘
```

### 3. Refined Dataset Viewer

```
┌─────────────────────────────────────────────────────────────────────┐
│ AI Image Processing Dashboard                              [User ▼] │
├─────────────────────────────────────────────────────────────────────┤
│ │ Categories │ Support Set │ Refined Dataset │ Analytics │ Settings │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │ Category: Category 1           Classification Method: [Method 1▼]│ │
│  │                                                                │ │
│  │ [Search...]          Sort by: [Confidence▼]     View: [Grid▼]  │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                                                                │ │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐      │ │
│  │  │        │ │        │ │        │ │        │ │        │      │ │
│  │  │ Class A │ │ Class A │ │ Class A │ │ Class B │ │ Class B │      │ │
│  │  │  95%    │ │  92%    │ │  87%    │ │  98%    │ │  91%    │      │ │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘      │ │
│  │                                                                │ │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐      │ │
│  │  │        │ │        │ │        │ │        │ │        │      │ │
│  │  │ Class B │ │ Class C │ │ Class C │ │ Class C │ │ Class D │      │ │
│  │  │  89%    │ │  94%    │ │  90%    │ │  85%    │ │  97%    │      │ │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘      │ │
│  │                                                                │ │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐      │ │
│  │  │        │ │        │ │        │ │        │ │        │      │ │
│  │  │ Class D │ │ Class D │ │ Class A │ │ Class B │ │ Class A │      │ │
│  │  │  82%    │ │  88%    │ │  93%    │ │  96%    │ │  92%    │      │ │
│  │  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘      │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌────────────────────────────┐  ┌────────────────────────────────┐ │
│  │ Selected: 0 items          │  │ Class Distribution             │ │
│  │                            │  │ ┌──────────────────────────┐   │ │
│  │ Actions:                   │  │ │                          │   │ │
│  │ [Reclassify▼] [Export▼]    │  │ │        [BAR CHART]       │   │ │
│  │                            │  │ │                          │   │ │
│  │ [Apply Different Method▼]  │  │ └──────────────────────────┘   │ │
│  └────────────────────────────┘  └────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 4. Comparison View

```
┌─────────────────────────────────────────────────────────────────────┐
│ AI Image Processing Dashboard                              [User ▼] │
├─────────────────────────────────────────────────────────────────────┤
│ │ Categories │ Support Set │ Refined Dataset │ Analytics │ Settings │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Comparison View: Support Set vs. Refined Dataset               │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌─────────────────────────────┐  ┌─────────────────────────────┐   │
│  │ Support Set                 │  │ Refined Dataset             │   │
│  │ Class: [Class A▼]           │  │ Method: [Method 1▼]         │   │
│  │                             │  │                             │   │
│  │  ┌────────┐ ┌────────┐     │  │  ┌────────┐ ┌────────┐     │   │
│  │  │        │ │        │     │  │  │        │ │        │     │   │
│  │  │ Example│ │ Example│     │  │  │ Result │ │ Result │     │   │
│  │  │   1    │ │   2    │     │  │  │   1    │ │   2    │     │   │
│  │  └────────┘ └────────┘     │  │  └────────┘ └────────┘     │   │
│  │                             │  │                             │   │
│  │  ┌────────┐ ┌────────┐     │  │  ┌────────┐ ┌────────┐     │   │
│  │  │        │ │        │     │  │  │        │ │        │     │   │
│  │  │ Example│ │ Example│     │  │  │ Result │ │ Result │     │   │
│  │  │   3    │ │   4    │     │  │  │   3    │ │   4    │     │   │
│  │  └────────┘ └────────┘     │  │  └────────┘ └────────┘     │   │
│  │                             │  │                             │   │
│  └─────────────────────────────┘  └─────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Selected Example: 1            Selected Result: 1               │ │
│  │ ┌─────────────────────┐        ┌─────────────────────┐         │ │
│  │ │                     │        │                     │         │ │
│  │ │                     │        │                     │         │ │
│  │ │       Image         │        │       Image         │         │ │
│  │ │      Preview        │        │      Preview        │         │ │
│  │ │                     │        │                     │         │ │
│  │ │                     │        │                     │         │ │
│  │ └─────────────────────┘        └─────────────────────┘         │ │
│  │                                                                 │ │
│  │ Details:                        Details:                        │ │
│  │ • ID: support_123              • ID: result_456                │ │
│  │ • Size: 640x480                • Size: 320x240                 │ │
│  │ • Date: 2023-06-15             • Date: 2023-06-18              │ │
│  │ • Metadata: person,outdoor     • Confidence: 94%               │ │
│  │                                • Class: Class A                │ │
│  │                                                                 │ │
│  │ Actions: [Use as Example]      Actions: [Reclassify] [Approve] │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 5. Analytics Dashboard

```
┌─────────────────────────────────────────────────────────────────────┐
│ AI Image Processing Dashboard                              [User ▼] │
├─────────────────────────────────────────────────────────────────────┤
│ │ Categories │ Support Set │ Refined Dataset │ Analytics │ Settings │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Analytics: Category 1                    Period: [Last 30 days▼]│ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌─────────────────────────────┐  ┌─────────────────────────────┐   │
│  │ Class Distribution          │  │ Classification Confidence    │   │
│  │ ┌───────────────────────┐  │  │ ┌───────────────────────┐   │   │
│  │ │                       │  │  │ │                       │   │   │
│  │ │                       │  │  │ │                       │   │   │
│  │ │       PIE CHART       │  │  │ │      BAR CHART        │   │   │
│  │ │                       │  │  │ │                       │   │   │
│  │ │                       │  │  │ │                       │   │   │
│  │ └───────────────────────┘  │  │ └───────────────────────┘   │   │
│  │                             │  │                             │   │
│  │ Class A: 35%                │  │ Method 1: 92% avg confidence│   │
│  │ Class B: 25%                │  │ Method 2: 87% avg confidence│   │
│  │ Class C: 20%                │  │ Method 3: 89% avg confidence│   │
│  │ Class D: 20%                │  │ Method 4: 85% avg confidence│   │
│  └─────────────────────────────┘  └─────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────┐  ┌─────────────────────────────┐   │
│  │ Processing History          │  │ Method Comparison            │   │
│  │ ┌───────────────────────┐  │  │ ┌───────────────────────┐   │   │
│  │ │                       │  │  │ │                       │   │   │
│  │ │                       │  │  │ │                       │   │   │
│  │ │       LINE CHART      │  │  │ │    STACKED BARS       │   │   │
│  │ │                       │  │  │ │                       │   │   │
│  │ │                       │  │  │ │                       │   │   │
│  │ └───────────────────────┘  │  │ └───────────────────────┘   │   │
│  │                             │  │                             │   │
│  │ Total images: 120           │  │ Accuracy:                   │   │
│  │ Last processed: 2023-06-20  │  │ • Method 1: 92%             │   │
│  │ Avg. processing time: 1.2s  │  │ • Method 2: 88%             │   │
│  │                             │  │ • Method 3: 90%             │   │
│  └─────────────────────────────┘  └─────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ Export Options:                                                 │ │
│  │ [Export Data CSV] [Export Charts PDF] [Schedule Regular Reports]│ │
│  └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Structure

### Frontend Components

1. **Layout Components**
   - MainLayout (overall page structure)
   - Navbar (navigation)
   - Sidebar (context-specific options)
   - Footer

2. **View Components**
   - DashboardView (home screen)
   - SupportSetView (support set browser)
   - RefinedDatasetView (refined dataset viewer)
   - ComparisonView (side-by-side comparison)
   - AnalyticsView (charts and statistics)
   - SettingsView (configuration)

3. **UI Components**
   - ImageCard (display image with metadata)
   - ImageGallery (grid of images)
   - ClassSelector (select/filter by class)
   - MethodSelector (select classification method)
   - FileUploader (upload new images)
   - DragDropTarget (for reclassification)
   - FilterControls (sorting and filtering)
   - Pagination (navigate through results)

4. **Visualization Components**
   - ClassDistributionChart (pie chart)
   - ConfidenceChart (bar chart)
   - TimelineChart (line chart)
   - MethodComparisonChart (stacked bars)
   - StatCard (display key statistics)

### Backend Components

1. **API Endpoints**
   - Category endpoints (list, details)
   - Support Set endpoints (list, CRUD)
   - Refined Dataset endpoints (list, reclassify)
   - Metadata endpoints (statistics, analytics)
   - Classification endpoints (apply methods)
   - File management endpoints (upload, organize)

2. **Services**
   - ClassificationService (apply classification methods)
   - MetadataService (manage image metadata)
   - FileService (file system operations)
   - AnalyticsService (generate statistics and charts)
   - UserService (authentication and preferences)

3. **Data Models**
   - Category (metadata about a category)
   - SupportSet (collection of example images)
   - RefinedDataset (classified results)
   - ImageMetadata (information about an image)
   - ClassificationResult (output of classification)
   - UserPreferences (user-specific settings)

## Interaction Patterns

1. **Drag and Drop Reclassification**
   - User drags image from gallery
   - Drops onto target class
   - System updates classification
   - UI refreshes to show new organization

2. **Batch Operations**
   - User selects multiple images
   - Chooses action (reclassify, delete, export)
   - Confirms operation
   - System processes in background
   - Real-time progress updates via WebSocket

3. **Method Comparison**
   - User selects different methods
   - Views side-by-side results
   - Can approve/reject individual classifications
   - Analytics update to show comparative performance

4. **Support Set Management**
   - User uploads new examples to support set
   - Organizes examples into classes
   - System automatically updates classifications
   - User can view impact on refined dataset

## Responsive Design Considerations

1. **Desktop (>1200px)**
   - Full layout with sidebars and detailed information
   - Multi-column galleries and charts
   - Advanced filtering and batch operations

2. **Tablet (768px-1199px)**
   - Condensed layout with collapsible panels
   - Reduced columns in galleries
   - Simplified filtering options

3. **Mobile (<767px)**
   - Single column layout
   - Stacked panels with expand/collapse
   - Essential controls only
   - Touch-optimized interactions 