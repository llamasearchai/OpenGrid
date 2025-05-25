#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <spdlog/spdlog.h>

// Main entry point for OpenGrid Qt/QML Desktop Application
int main(int argc, char *argv[]) {
    spdlog::info("Starting OpenGrid Application...");

    QGuiApplication app(argc, argv); // Use QGuiApplication for QML-heavy apps

    QQmlApplicationEngine engine;
    // Define the URL to your main QML file
    // It could be a Qt resource (qrc) or a local file path during development
    const QUrl url(QStringLiteral("qrc:/opengrid/qml/main.qml")); 
    // Or for local file: const QUrl url(QUrl::fromLocalFile("../visualization/qml/main.qml"));

    QObject::connect(&engine, &QQmlApplicationEngine::objectCreationFailed,
                     &app, []() { 
                        spdlog::critical("QML object creation failed, exiting.");
                        QCoreApplication::exit(-1); 
                     }, Qt::QueuedConnection);
    engine.load(url);

    if (engine.rootObjects().isEmpty()) {
        spdlog::critical("Failed to load QML: rootObjects is empty. Ensure main.qml exists and is valid.");
        return -1;
    }

    spdlog::info("OpenGrid QML Engine initialized and main.qml loaded.");
    int exec_result = app.exec();
    spdlog::info("OpenGrid Application finished with exit code: {}", exec_result);
    return exec_result;
} 