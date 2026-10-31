#!/usr/bin/env ruby
require 'xcodeproj'

class SplashProject < Xcodeproj::Project
  def generate_uuid
    @next_uuid = (@next_uuid || 0) + 1
    format('53504C53%016X', @next_uuid)
  end
end

root = File.expand_path(__dir__)
path = File.join(root, 'Splash.xcodeproj')
project = SplashProject.new(path)
project.root_object.attributes['LastUpgradeCheck'] = '2700'
project.root_object.attributes['TargetAttributes'] = {}

app = project.new_target(:application, 'Splash', :osx, '26.4')
tests = project.new_target(:unit_test_bundle, 'SplashTests', :osx, '26.4')
tests.add_dependency(app)
app.resources_build_phase.add_file_reference(project.main_group.new_file('model_tools.py'))
manifest = File.join(root, '.build/runtime-manifest.json')
app.resources_build_phase.add_file_reference(project.main_group.new_file(manifest)) if File.file?(manifest)
{'Splash' => app, 'SplashTests' => tests}.each do |directory, target|
  group = project.main_group.new_group(directory, directory)
  Dir.glob(File.join(root, directory, '*.swift')).sort.each do |source|
    ref = group.new_file(File.basename(source))
    target.source_build_phase.add_file_reference(ref)
  end
  target.build_configurations.each do |config|
    config.build_settings['SWIFT_VERSION'] = '5.0'
    config.build_settings['MACOSX_DEPLOYMENT_TARGET'] = '26.4'
    config.build_settings['SDKROOT'] = 'macosx'
    config.build_settings['CODE_SIGNING_ALLOWED'] = 'NO'
    config.build_settings['GENERATE_INFOPLIST_FILE'] = 'NO'
    config.build_settings['MARKETING_VERSION'] = '0.1.0'
    config.build_settings['CURRENT_PROJECT_VERSION'] = '1'
    if target == app
      config.build_settings['SPLASH_RELEASE_VERSION'] = '0.1.0-rc.1'
      config.build_settings['PRODUCT_BUNDLE_IDENTIFIER'] = 'org.splashml.desktop'
      config.build_settings['INFOPLIST_FILE'] = 'Splash/Info.plist'
      config.build_settings['ASSETCATALOG_COMPILER_APPICON_NAME'] = 'AppIcon'
      config.build_settings['ENABLE_HARDENED_RUNTIME'] = 'YES'
    else
      config.build_settings['PRODUCT_BUNDLE_IDENTIFIER'] = 'org.splashml.desktop.tests'
      config.build_settings['INFOPLIST_FILE'] = 'SplashTests/Info.plist'
      config.build_settings['TEST_HOST'] = '$(BUILT_PRODUCTS_DIR)/Splash.app/Contents/MacOS/Splash'
      config.build_settings['BUNDLE_LOADER'] = '$(TEST_HOST)'
    end
  end
end

app.resources_build_phase.add_file_reference(project.main_group.new_file('Splash/Assets.xcassets'))

scheme = Xcodeproj::XCScheme.new
scheme.add_build_target(app)
scheme.add_test_target(tests)
scheme.save_as(path, 'Splash', true)
project.save
