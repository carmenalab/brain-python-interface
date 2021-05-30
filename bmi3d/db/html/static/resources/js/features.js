// Use "require" only if run from command line
if (typeof(require) !== 'undefined') {
    var jsdom = require('jsdom');
    const { JSDOM } = jsdom;
    const dom = new JSDOM(`    <div id="features" hidden="true">
        <ul>
            <li><input type="checkbox" id="feat_feature1" value="feature1" name="feature1">
                <label for="feat_feature1">Feature 1</label>
            </li>
            <li><input type="checkbox" id="feat_feature2" value="feature2" name="feature2">
                <label for="feat_feature2">Feature 2</label></li>
        </ul>
    </div>`);
    var document = dom.window.document;
    var $ = jQuery = require('jquery')(dom.window);
}

function Features() {
    this.features_obj = $("#features");
}
/*
 * Uncheck all the features
 */
Features.prototype.clear = function() {
    $("#features input[type=checkbox]").each(
        function() {
            this.checked = false;
       }
    );
}
/*
 * Check boxes for a set of features
 */
Features.prototype.select_features = function(info_feats) {
    // set checkmarks for all the features specified in the 'info'
    $("#features input[type=checkbox]").each(
        function() {
            this.checked = false;
            for (var idx in info_feats) {
                if (this.name == info_feats[idx])
                    this.checked = true;
            }
       }
    );
}
/*
 * Get a JSON object with the list of enabled features. Keys = feature names,
 * values are 'true'
 */
Features.prototype.get_checked_features = function () {
    var feats = {};
    $("#features input").each(function() {
        if (this.checked)
            feats[this.name] = this.checked;
    });
    return feats;
}
/*
 * Disable the ability to check/uncheck features
 */
Features.prototype.disable_entry = function () {
    $("#features input").attr("disabled", "disabled");
}
/*
 * Enable the ability to check features
 */
Features.prototype.enable_entry = function() {
    $("#features input").removeAttr("disabled");
}
/*
 * If any of the features are chagned or unchanged, run the callback
 */
Features.prototype.bind_change_callback = function(callback) {
    $("#features input").change(callback);
}
/*
 * Disable any callbacks assigned by 'bind_change_callback'
 */
Features.prototype.unbind_change_callback = function() {
    $("#features input").unbind("change");
}

if (typeof(module) !== 'undefined' && module.exports) {
  exports.Features = Features;
  exports.$ = $;
}
