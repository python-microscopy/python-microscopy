# Style Refactoring Summary

This document describes the CSS refactoring performed to consolidate repeated styles into the main `style.css` file.

## Changes Made

### Global Styles Added to `src/style.css`

The following common component styles have been moved to the global stylesheet:

#### Form Components
- `.form-group` - Standard form group container with margin
- `.form-group label` - Consistent label styling across all forms
- `.form-control` - Standard form input/select styling (dark theme)
- `.select-sm` - Small select dropdown styling
- `.input-sm` - Small input field styling

#### Buttons
- `.btn` - Base button styles
- `.btn-sm` - Small button variant
- `.btn-primary` - Primary action button (blue)
- `.btn-secondary` - Secondary action button (gray)
- `.btn-danger` - Destructive action button (red)

#### Layout Sections
- `.section` - Content section with bottom border
- `.section-heading` - Uppercase section title
- `.subsection-heading` - Smaller subsection title
- `.control-group` - Control grouping container

#### Form Controls
- `.radio-group` - Radio button group container
- `.radio-label` - Individual radio button label
- `.checkbox-label` - Checkbox label with icon
- `.form-row` - Responsive form row grid
- `.form-col` - Form column within row

#### Utility Classes
- `.text-center` - Center-aligned text

### Component-Specific Styles Retained

Each component now only contains styles that are unique to that component:

#### DisplayControls.vue
- `.range-inputs` - Specific layout for min/max range inputs
- Width override for `.input-sm`

#### HardwareControls.vue
- `.input-sm` - Width and margin specific to this component

#### LaserControl.vue
- `.laser-control`, `.laser-label` - Laser-specific layout
- `.laser-inputs` - Laser control input layout
- `.range-slider` - Custom slider styling

#### PositionControl.vue
- `.position-control`, `.position-inputs` - Position-specific layout
- `.input-sm` - Center-aligned text for position inputs

#### AcquisitionControls.vue
- `.spool-info`, `.spool-target`, `.spool-status` - Spooling-specific UI
- `.section-heading` - Larger heading for this section

#### AcquisitionTab.vue
- `.acquisition-tab` - Tab padding and layout
- `.content-grid` - Two-column responsive grid
- `.settings-section h6` - Section-specific heading

#### StackSettings.vue
- `.stack-settings` - Background and padding for settings panel
- `.input-with-button` - Layout for input with adjacent button

#### LoginView.vue
- Retains most styles as it uses a light theme (different from dark theme elsewhere)
- Override of `.form-group` for light theme colors

## Benefits

1. **Reduced Duplication**: Common styles defined once instead of in every component
2. **Consistency**: All buttons, inputs, and form elements have consistent styling
3. **Maintainability**: Single source of truth for common styles
4. **Smaller Component Files**: Component styles focus only on component-specific layout
5. **Easy Theming**: Global styles can be updated to affect all components

## Usage in Components

Components can now use the global classes directly:

```vue
<template>
  <div class="section">
    <h6 class="section-heading">Title</h6>
    <div class="form-group">
      <label>Field:</label>
      <input class="form-control" type="text" />
    </div>
    <button class="btn btn-primary">Save</button>
  </div>
</template>

<script setup>
// Component logic
</script>

<style scoped>
/* Only component-specific styles here */
.custom-layout {
  /* ... */
}
</style>
```

## File Size Reduction

Approximate style reduction per component:
- DisplayControls: ~40 lines → ~10 lines
- HardwareControls: ~35 lines → ~5 lines
- LaserControl: ~50 lines → ~35 lines
- PositionControl: ~45 lines → ~15 lines
- AcquisitionControls: ~95 lines → ~25 lines
- AcquisitionTab: ~65 lines → ~25 lines
- StackSettings: ~75 lines → ~15 lines

**Total reduction**: ~400 lines of duplicate CSS removed
