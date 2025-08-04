#!/usr/bin/env node

/**
 * Qubots MCP Tools
 * Model Context Protocol tools for optimization workflows
 */

const { Command } = require('commander');
const chalk = require('chalk');
const ora = require('ora');
const fs = require('fs-extra');
const path = require('path');
const axios = require('axios');

const program = new Command();

// Configuration
const CONFIG = {
  apiUrl: process.env.QUBOTS_API_URL || 'http://localhost:8000',
  giteaUrl: process.env.QUBOTS_GITEA_URL || 'http://localhost:3000',
  outputDir: process.env.QUBOTS_OUTPUT_DIR || './qubots_output'
};

// Ensure output directory exists
fs.ensureDirSync(CONFIG.outputDir);

program
  .name('qubots-mcp')
  .description('MCP tools for Qubots optimization workflows')
  .version('1.0.0');

/**
 * Create a new workflow
 */
program
  .command('workflow-create')
  .description('Create a new optimization workflow')
  .option('-n, --name <name>', 'Workflow name')
  .option('-d, --description <description>', 'Workflow description')
  .option('-o, --output <file>', 'Output file path')
  .option('--interactive', 'Interactive mode')
  .action(async (options) => {
    const spinner = ora('Creating workflow...').start();
    
    try {
      let workflowData = {
        name: options.name || 'New Workflow',
        description: options.description || 'A new optimization workflow',
        version: '1.0.0',
        author: 'Qubots User',
        created_at: new Date().toISOString(),
        nodes: [],
        edges: [],
        global_parameters: {},
        metadata: {}
      };

      if (options.interactive) {
        const inquirer = require('inquirer');
        
        const answers = await inquirer.prompt([
          {
            type: 'input',
            name: 'name',
            message: 'Workflow name:',
            default: workflowData.name
          },
          {
            type: 'input',
            name: 'description',
            message: 'Workflow description:',
            default: workflowData.description
          },
          {
            type: 'input',
            name: 'author',
            message: 'Author:',
            default: workflowData.author
          }
        ]);
        
        workflowData = { ...workflowData, ...answers };
      }

      const outputFile = options.output || path.join(CONFIG.outputDir, `${workflowData.name.toLowerCase().replace(/\s+/g, '-')}.json`);
      
      await fs.writeJson(outputFile, workflowData, { spaces: 2 });
      
      spinner.succeed(`Workflow created: ${chalk.green(outputFile)}`);
      
      console.log(chalk.blue('\nNext steps:'));
      console.log('1. Add components to your workflow');
      console.log('2. Configure parameters');
      console.log('3. Validate with: qubots-mcp workflow-validate');
      console.log('4. Generate code with: qubots-mcp code-generate');
      
    } catch (error) {
      spinner.fail(`Failed to create workflow: ${error.message}`);
      process.exit(1);
    }
  });

/**
 * Validate a workflow
 */
program
  .command('workflow-validate')
  .description('Validate a workflow definition')
  .argument('<file>', 'Workflow file to validate')
  .option('--detailed', 'Show detailed validation results')
  .action(async (file, options) => {
    const spinner = ora('Validating workflow...').start();
    
    try {
      const workflowData = await fs.readJson(file);
      
      const response = await axios.post(`${CONFIG.apiUrl}/api/workflows/validate`, workflowData);
      const validation = response.data;
      
      if (validation.valid) {
        spinner.succeed('Workflow is valid!');
        
        if (validation.warnings && validation.warnings.length > 0) {
          console.log(chalk.yellow('\nWarnings:'));
          validation.warnings.forEach(warning => {
            console.log(chalk.yellow(`  ⚠️  ${warning}`));
          });
        }
      } else {
        spinner.fail('Workflow validation failed');
        
        console.log(chalk.red('\nErrors:'));
        validation.errors.forEach(error => {
          console.log(chalk.red(`  ❌ ${error}`));
        });
      }
      
      if (options.detailed) {
        console.log(chalk.blue('\nDetailed Results:'));
        console.log(JSON.stringify(validation, null, 2));
      }
      
    } catch (error) {
      spinner.fail(`Validation failed: ${error.message}`);
      process.exit(1);
    }
  });

/**
 * Execute a workflow
 */
program
  .command('workflow-execute')
  .description('Execute an optimization workflow')
  .argument('<file>', 'Workflow file to execute')
  .option('--timeout <seconds>', 'Execution timeout', '300')
  .option('--output <file>', 'Save results to file')
  .action(async (file, options) => {
    const spinner = ora('Executing workflow...').start();
    
    try {
      const workflowData = await fs.readJson(file);
      
      const response = await axios.post(`${CONFIG.apiUrl}/api/workflows/execute`, {
        workflow: workflowData,
        parameters: {}
      }, {
        timeout: parseInt(options.timeout) * 1000
      });
      
      const result = response.data;
      
      spinner.succeed(`Workflow executed successfully in ${result.runtime_seconds}s`);
      
      console.log(chalk.green('\nResults:'));
      console.log(`  Best value: ${result.best_value}`);
      console.log(`  Status: ${result.termination_reason}`);
      console.log(`  Iterations: ${result.iterations}`);
      
      if (options.output) {
        await fs.writeJson(options.output, result, { spaces: 2 });
        console.log(chalk.blue(`\nResults saved to: ${options.output}`));
      }
      
    } catch (error) {
      spinner.fail(`Execution failed: ${error.message}`);
      process.exit(1);
    }
  });

/**
 * Search for components
 */
program
  .command('component-search')
  .description('Search for optimization components')
  .option('-t, --type <type>', 'Component type (problem, optimizer)')
  .option('-d, --domain <domain>', 'Problem domain')
  .option('-q, --query <query>', 'Search query')
  .option('--limit <number>', 'Maximum results', '10')
  .action(async (options) => {
    const spinner = ora('Searching components...').start();
    
    try {
      const params = new URLSearchParams();
      if (options.type) params.append('type', options.type);
      if (options.domain) params.append('domain', options.domain);
      if (options.query) params.append('search', options.query);
      
      const response = await axios.get(`${CONFIG.apiUrl}/api/components?${params}`);
      const components = response.data.slice(0, parseInt(options.limit));
      
      spinner.succeed(`Found ${components.length} components`);
      
      if (components.length === 0) {
        console.log(chalk.yellow('No components found matching your criteria.'));
        return;
      }
      
      console.log(chalk.blue('\nComponents:'));
      components.forEach((comp, index) => {
        console.log(`\n${index + 1}. ${chalk.green(comp.name)}`);
        console.log(`   Type: ${comp.type}`);
        console.log(`   Domain: ${comp.domain}`);
        console.log(`   Description: ${comp.description}`);
        console.log(`   Repository: ${comp.repository}`);
        console.log(`   Rating: ${comp.rating} ⭐ (${comp.downloads} downloads)`);
      });
      
    } catch (error) {
      spinner.fail(`Search failed: ${error.message}`);
      process.exit(1);
    }
  });

/**
 * Install a component
 */
program
  .command('component-install')
  .description('Install an optimization component')
  .argument('<component-id>', 'Component ID to install')
  .action(async (componentId) => {
    const spinner = ora(`Installing ${componentId}...`).start();
    
    try {
      const response = await axios.post(`${CONFIG.apiUrl}/api/components/${componentId}/install`);
      
      spinner.succeed(`Component ${componentId} installed successfully!`);
      console.log(chalk.green(response.data.message));
      
    } catch (error) {
      spinner.fail(`Installation failed: ${error.message}`);
      process.exit(1);
    }
  });

/**
 * Generate code from workflow
 */
program
  .command('code-generate')
  .description('Generate Python code from workflow')
  .argument('<file>', 'Workflow file')
  .option('-o, --output <file>', 'Output Python file')
  .option('--format <format>', 'Output format (python, json)', 'python')
  .action(async (file, options) => {
    const spinner = ora('Generating code...').start();
    
    try {
      const workflowData = await fs.readJson(file);
      
      const response = await axios.post(`${CONFIG.apiUrl}/api/workflows/generate-code`, workflowData);
      const result = response.data;
      
      const outputFile = options.output || path.join(CONFIG.outputDir, result.filename);
      
      if (options.format === 'python') {
        await fs.writeFile(outputFile, result.code);
        spinner.succeed(`Python code generated: ${chalk.green(outputFile)}`);
      } else if (options.format === 'json') {
        const mcpData = {
          workflow: workflowData,
          generated_code: result.code,
          timestamp: new Date().toISOString()
        };
        await fs.writeJson(outputFile.replace('.py', '.json'), mcpData, { spaces: 2 });
        spinner.succeed(`JSON export generated: ${chalk.green(outputFile.replace('.py', '.json'))}`);
      }
      
      console.log(chalk.blue('\nGenerated files:'));
      console.log(`  Code: ${outputFile}`);
      console.log(`  Language: ${result.language}`);
      
    } catch (error) {
      spinner.fail(`Code generation failed: ${error.message}`);
      process.exit(1);
    }
  });

/**
 * Show system status
 */
program
  .command('status')
  .description('Show Qubots system status')
  .action(async () => {
    const spinner = ora('Checking system status...').start();
    
    try {
      const [apiResponse, giteaResponse] = await Promise.allSettled([
        axios.get(`${CONFIG.apiUrl}/api/system/status`),
        axios.get(`${CONFIG.giteaUrl}/api/healthz`)
      ]);
      
      spinner.stop();
      
      console.log(chalk.blue('🔧 Qubots System Status'));
      console.log('=' * 30);
      
      // API Status
      if (apiResponse.status === 'fulfilled') {
        const status = apiResponse.value.data;
        console.log(`${chalk.green('✅')} API: Connected`);
        console.log(`   Profile: ${status.profile}`);
        console.log(`   Mode: ${status.local_mode ? 'Local' : 'Cloud'}`);
        console.log(`   Authenticated: ${status.authenticated ? 'Yes' : 'No'}`);
      } else {
        console.log(`${chalk.red('❌')} API: Disconnected`);
      }
      
      // Gitea Status
      if (giteaResponse.status === 'fulfilled') {
        console.log(`${chalk.green('✅')} Gitea: Connected`);
      } else {
        console.log(`${chalk.red('❌')} Gitea: Disconnected`);
      }
      
      console.log(`\n📁 Output Directory: ${CONFIG.outputDir}`);
      
    } catch (error) {
      spinner.fail(`Status check failed: ${error.message}`);
    }
  });

// Parse command line arguments
program.parse();

// If no command provided, show help
if (!process.argv.slice(2).length) {
  program.outputHelp();
}
