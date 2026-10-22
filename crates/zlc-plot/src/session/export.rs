//! The export session of a saved figure (`figure_artifact._export`): a fresh
//! session built from the panel's recipe at the export scale, so the image
//! is what reopening the archive draws.

use crate::configuration::data_contract::PlotSchema;
use crate::configuration::error::{PlotError, Result};
use crate::configuration::parameters::LIMIT_PAIRS;
use crate::configuration::resolve::{resolve_configuration, ConfigurationUpdates, OverlayHandle, ResolveRequest};
use crate::configuration::selectors::thresholds_wire;
use crate::configuration::spec::PlotSpec;
use crate::configuration::state::{InteractionUpdate, PresentationUpdate};
use crate::configuration::value::{Value, ValueMap};

use super::core::{PlotSession, SessionServices};
use super::preparation::DataPreparation;
use super::projected::PlotInput;

impl PlotSession {
    /// A fresh export session of this panel's accepted recipe.
    pub fn export_session(&self, services: &SessionServices) -> Result<PlotSession> {
        let updates = ConfigurationUpdates {
            size: Some(self.size().to_owned()),
            image_overlay: Some(self.image_overlay().map(|overlay| overlay as OverlayHandle)),
            viewport: Some(self.viewport()), classifier_thresholds: Some(thresholds_wire(&self.classifier_threshold_targets_state(true)?)),
            facet_focus: Some(self.facet_focus().map(|focus| focus as i128)), selectors: Some(self.configured_selectors()?),
            fit: Some(self.fit_configuration()),
            interaction: Some(InteractionUpdate { series_lock: Some(self.display_state().interaction.series_lock.clone()) }),
            presentation: Some(PresentationUpdate { series_readout: Some(self.renderer().series_presentation().series_readout) }),
            ..Default::default()
        };
        Self::for_export(self.data(), self.spec().clone(), self.display_state().values.clone(), updates, services)
    }

    /// `PlotSession(plot_input, spec, parameters=, _for_export=True,
    /// initial_configuration=)`: a fresh session of one recipe at the export
    /// scale, its other parts in `updates` (the fit drawn, not live).
    pub fn for_export(input: PlotInput, spec: PlotSpec, mut parameters: ValueMap, mut updates: ConfigurationUpdates,
                      services: &SessionServices) -> Result<PlotSession> {
        let schema = crate::configuration::display::parameter_schema_for(&spec, &services.defaults.style);
        let mut deferred = ValueMap::new();
        for (mode, low, high) in LIMIT_PAIRS {
            if !(schema.contains(mode) && schema.contains(low) && schema.contains(high)) { continue; }
            if parameters.get(mode) != Some(&Value::text("fixed")) { continue; }
            let (low_value, high_value) = (parameters.get(low).cloned().unwrap_or(Value::None), parameters.get(high).cloned().unwrap_or(Value::None));
            if !low_value.is_none() && !high_value.is_none() { continue; }
            deferred.insert(mode, Value::text("fixed"));
            deferred.insert(low, low_value);
            deferred.insert(high, high_value);
            for name in [mode, low, high] { parameters.remove(name); }
        }
        updates.parameters = Some(schema.initial_values(Some(&parameters))?);
        updates.fit_live = Some(false);
        let source = input.source().ok_or_else(|| PlotError::value("Pulse timelines are not data-backed Figure artifacts"))?;
        let candidate = resolve_configuration(ResolveRequest {
            schema: Some(PlotSchema::new(source.schema.clone())), spec: Some(spec), defaults: Some(services.defaults.clone()),
            fit: Some(services.fit.clone()), unit_registry: services.registry.clone(), configuration_updates: Some(updates), ..Default::default()
        })?;
        let mut schemas = crate::projection::SchemaCache::default();
        let prepared = DataPreparation::new(candidate.clone(), None).prepare(&input, None, &mut schemas, &services.settings(), &services.fit.engine)?;
        let layout = &services.defaults.layout;
        let mut export = PlotSession::build(prepared, candidate, layout.export_dpi / layout.live_dpi(), None, services.clone(), true)?;
        export.finish_presentation(true, None);
        if !deferred.is_empty() { export.configure_parameters(deferred)?; }
        Ok(export)
    }

    /// `save(path, restore_display=False)`'s image bytes in `format`.
    pub fn save_image(&mut self, format: &str) -> Result<Vec<u8>> {
        let dpi = self.services.defaults.layout.export_dpi;
        self.renderer.save(format, dpi, false)
    }
}
