# PYME Acquire Web UI - Modern Vue.js App

A modern, single-page Vue.js application for controlling PYME Acquire microscopy system via a web interface.

## Features

- **Modern Vue 3** with Composition API
- **Single-file components** (.vue files)
- **Pinia** for state management
- **Vue Router** for navigation
- **Vite** for fast development and optimized builds
- **Real-time updates** via long-polling
- **Responsive design** with dark theme
- **Client-side authentication**

## Project Structure

```
app/
├── index.html              # Main HTML entry point
├── package.json            # Dependencies and scripts
├── vite.config.js         # Vite configuration
└── src/
    ├── main.js            # Application entry point
    ├── App.vue            # Root component
    ├── style.css          # Global styles
    ├── router/
    │   └── index.js       # Router configuration
    ├── stores/            # Pinia stores (state management)
    │   ├── auth.js        # Authentication state
    │   ├── scope.js       # Microscope state
    │   ├── spooler.js     # Data acquisition state
    │   └── stack.js       # Stack settings state
    ├── api/
    │   └── index.js       # API client functions
    ├── views/             # Page-level components
    │   ├── LoginView.vue
    │   └── MainView.vue
    └── components/        # Reusable components
        ├── CameraDisplay.vue
        ├── DisplayControls.vue
        ├── HardwareControls.vue
        ├── AcquisitionControls.vue
        ├── LaserControl.vue
        ├── PositionControl.vue
        ├── StackSettings.vue
        └── tabs/
            ├── AcquisitionTab.vue
            ├── StateTab.vue
            └── ConsoleTab.vue
```

## Development

### Prerequisites

- Node.js 18+ and npm
- Python PYME Acquire server running on port 9797

### Installation

```bash
cd PYME/Acquire/webui/app
npm install
```

### Development Server

```bash
npm run dev
```

This starts the Vite development server on http://localhost:3000 with hot-module replacement. API requests are proxied to the Python server at http://localhost:9797.

### Building for Production

```bash
npm run build
```

This creates an optimized production build in `../dist/` directory.

### Preview Production Build

```bash
npm run preview
```

## Architecture

### State Management

The app uses **Pinia** stores for managing application state:

- **authStore**: User authentication and session management
- **scopeStore**: Microscope hardware state (cameras, lasers, positioning)
- **spoolerStore**: Data acquisition and spooling state
- **stackStore**: Z-stack acquisition settings

### API Communication

All server communication is handled through the `api/index.js` module which provides:

- RESTful API calls for configuration and control
- Long-polling for real-time state updates
- Cookie-based authentication

### Components

Components are organized by function:

- **Views**: Top-level page components (Login, Main)
- **Controls**: Hardware control components (Camera, Lasers, Positioning)
- **Tabs**: Tab content components (Acquisition, State, Console)
- **Shared**: Reusable UI components

### Routing

Vue Router handles navigation with authentication guards:

- `/login` - Login page
- `/` - Main application (requires authentication)

## Integration with Python Server

The app expects the Python PYME Acquire server to provide these endpoints:

### Authentication
- `GET /do_login?email=...&password=...` - Login and get token
- `GET /logout` - Logout
- `GET /api/user` - Get current user info

### Scope State
- `GET /get_scope_state` - Get current state
- `POST /update_scope_state` - Update state
- `GET /scope_state_longpoll` - Long-poll for state updates

### Camera
- `GET /get_frame_pzf` - Get camera frame (PZF format)

### Spooler
- `GET /spool_controller/info` - Get spooler info
- `POST /spool_controller/settings` - Update settings
- `GET /spool_controller/start_spooling` - Start acquisition
- `GET /spool_controller/stop_spooling` - Stop acquisition
- `GET /spool_controller/info_longpoll` - Long-poll for updates

### Stack Settings
- `GET /stack_settings/settings` - Get settings
- `POST /stack_settings/update` - Update settings
- `GET /stack_settings/settings_longpoll` - Long-poll for updates

## Deployment

To serve the built app from the Python server, the dist files should be served as static files. Update the Python server to serve the built `index.html` at the root path and static assets from the `dist/assets/` directory.

## Migration from Legacy App

This modern Vue app replaces the legacy jQuery/Vue 2 app with:

- ✅ Modern build tooling (Vite instead of manual scripts)
- ✅ TypeScript-ready architecture
- ✅ Proper component organization
- ✅ State management with Pinia
- ✅ Better developer experience with HMR
- ✅ Optimized production builds
- ✅ Maintainable codebase

## TODO

- [ ] Port PZF decoder for camera display
- [ ] Add protocol file selection UI
- [ ] Add simulation controls
- [ ] Implement camera frame autoscaling
- [ ] Add error boundaries and loading states
- [ ] Add unit tests
- [ ] Add E2E tests

## License

Same as PYME project (GPL v3)
